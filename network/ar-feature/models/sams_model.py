import argparse
import logging
from typing import Dict, List

import torch
from pytorch_lightning import Trainer
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data.dataloader import default_collate

from datasets.tryon_dataset import parse_num_channels, TryonDataset
from models.base_model import BaseModel
from models.networks.loss import VGGLoss, GANLoss
from models.networks.sams.sams_generator import SamsGenerator
from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
from options import gan_options
from visualization import tensor_list_for_board

logger = logging.getLogger("logger")


class SamsModel(BaseModel):
    """ Self Attentive Multi-Spade """

    @classmethod
    def modify_commandline_options(cls, parser: argparse.ArgumentParser, is_train):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser = super(SamsModel, cls).modify_commandline_options(parser, is_train)
        parser.set_defaults(person_inputs=("agnostic", "densepose", "flow"))
        parser.add_argument(
            "--encoder_input",
            default="flow",
            help="which of the --person_inputs to use as the encoder segmap input "
            "(only 1 allowed).",
        )
        # num previous frames fed as input = n_frames_total - 1
        parser.set_defaults(n_frames_total=5)
        # batch size effectively becomes n_frames_total * batch
        parser.set_defaults(batch_size=4)
        parser.add_argument(
            "--wt_l1",
            type=float,
            default=1.0,
            help="Weight applied to l1 loss in the generator",
        )
        parser.add_argument(
            "--wt_vgg",
            type=float,
            default=1.0,
            help="Weight applied to vgg loss in the generator",
        )
        parser.add_argument(
            "--wt_multiscale",
            type=float,
            default=1.0,
            help="Weight applied to adversarial multiscale loss in the generator",
        )
        parser.add_argument(
            "--wt_temporal",
            type=float,
            default=1.0,
            help="Weight applied to adversarial temporal loss in the generator",
        )
        parser.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )
        from models import networks

        parser = networks.modify_commandline_options(parser, is_train)
        parser = gan_options.modify_commandline_options(parser, is_train)
        return parser

    @staticmethod
    def apply_default_encoder_input(opt):
        """ Call in Base Options after opt parsed """
        if hasattr(opt, "encoder_input") and opt.encoder_input is None:
            opt.encoder_input = opt.person_inputs[0]
        return opt

    def __init__(self, hparams):
        # Lightning bug, see https://github.com/PyTorchLightning/pytorch-lightning/issues/924#issuecomment-673137383
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        super().__init__(hparams)
        self.n_frames_total = hparams.n_frames_total
        self.n_frames_now = (
            hparams.n_frames_now if hparams.n_frames_now else self.n_frames_total
        )
        self.inputs = hparams.person_inputs + hparams.cloth_inputs
        self.generator = SamsGenerator(hparams)
        self.resample = Resample2d()

        if self.is_train:
            init = hparams.init_type, hparams.init_variance
            self.generator.init_weights(*init)

            from models.networks import MultiscaleDiscriminator

            self.multiscale_discriminator = MultiscaleDiscriminator(hparams)
            self.multiscale_discriminator.init_weights(*init)

            enc_ch = parse_num_channels(hparams.encoder_input)
            temporal_in_channels = self.n_frames_total * (
                enc_ch + TryonDataset.RGB_CHANNELS
            )
            from models.networks import NLayerDiscriminator

            self.temporal_discriminator = NLayerDiscriminator(
                hparams, in_channels=temporal_in_channels
            )
            self.temporal_discriminator.init_weights(*init)

            self.criterion_GAN = GANLoss(hparams.gan_mode)
            self.criterion_l1 = L1Loss()
            self.criterion_VGG = VGGLoss()

            self.wt_l1 = hparams.wt_l1
            self.wt_vgg = hparams.wt_vgg
            self.wt_multiscale = hparams.wt_multiscale
            self.wt_temporal = hparams.wt_temporal

    def forward(self, *args, **kwargs):
        self.generator(*args, **kwargs)

    def configure_optimizers(self):
        # must do individual optimizers and schedulers per each network
        optimizer_g = Adam(self.generator.parameters(), self.hparams.lr)
        optimizer_d_multi = Adam(
            self.multiscale_discriminator.parameters(), self.hparams.lr_D
        )
        optimizer_d_temporal = Adam(
            self.temporal_discriminator.parameters(), self.hparams.lr_D
        )
        scheduler_g = self._make_step_scheduler(optimizer_g)
        scheduler_d_multi = self._make_step_scheduler(optimizer_d_multi)
        scheduler_d_temporal = self._make_step_scheduler(optimizer_d_temporal)
        return (
            [optimizer_g, optimizer_d_multi, optimizer_d_temporal],
            [scheduler_g, scheduler_d_multi, scheduler_d_temporal],
        )

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx == 0:
            result = self.generator_step(batch)
        elif optimizer_idx == 1:
            result = self.multiscale_discriminator_step(batch)
        else:
            result = self.temporal_discriminator_step(batch)
            if self.global_step % self.hparams.display_count == 0:
                self.visualize(batch)

        return result

    def validation_step(self, batch, idx) -> Dict[str, Tensor]:
        self.batch = batch
        result = self.generator_step(batch, val=True)
        result['global_step'] = self.global_step

        return result

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass

    def generator_step(self, batch, val=False):
        loss_G_adv_multiscale = (  # also calls generator forward
            self.multiscale_adversarial_loss(batch, for_discriminator=False)
            * self.wt_multiscale
        )
        loss_G_adv_temporal = (
            self.temporal_adversarial_loss(batch, for_discriminator=False)
            * self.wt_temporal
        )
        ground_truth = batch["image"][:, -1, :, :, :]
        fake_frame = self.all_gen_frames[:, -1, :, :, :]
        loss_G_l1 = self.criterion_l1(fake_frame, ground_truth) * self.wt_l1
        loss_G_vgg = self.criterion_VGG(fake_frame, ground_truth) * self.wt_vgg

        loss_G = loss_G_l1 + loss_G_vgg + loss_G_adv_multiscale + loss_G_adv_temporal

        # Log
        val_ = "val_" if val else ""
        result = {'loss': loss_G}
        result[f"{val_}loss/G/adv_multiscale"] = loss_G_adv_multiscale
        result[f"{val_}loss/G/adv_temporal"] = loss_G_adv_temporal
        result[f"{val_}loss/G/l1+vgg"] = loss_G_l1 + loss_G_vgg
        result[f"{val_}loss/G/l1"] = loss_G_l1
        result[f"{val_}loss/G/vgg"] = loss_G_vgg
        return result

    def generate_n_frames(self, batch):
        # each Tensor is (b x N-Frames x c x h x w)
        labelmap: Dict[str, Tensor] = {key: batch[key] for key in self.inputs}

        # make a buffer of previous frames, also (b x N x c x h x w)
        all_generated_frames: Tensor = torch.zeros_like(batch["image"])
        flows = torch.unbind(batch["flow"], dim=1) if self.hparams.flow_warp else None

        # generate previous frames before this one.
        #   for progressive training, just generate from here
        start_idx = self.n_frames_total - self.n_frames_now
        for fIdx in range(start_idx, self.n_frames_total):
            # Prepare data...
            # all the guidance for the current frame
            weight_boundary = TryonDataset.RGB_CHANNELS
            labelmaps_this_frame: Dict[str, Tensor] = {
                name: lblmap[:, fIdx, :, :, :] for name, lblmap in labelmap.items()
            }
            prev_n_frames_G, prev_n_labelmaps = self.get_prev_frames_and_maps(
                batch, fIdx, all_generated_frames
            )
            # synthesize
            out: Tensor = self.generator.forward(
                prev_n_frames_G, prev_n_labelmaps, labelmaps_this_frame
            )
            fake_frame = out[:, :weight_boundary, :, :].clone()
            weight_mask = out[:, weight_boundary:, :, :].clone()

            if self.hparams.flow_warp:
                last_real_image = batch["image"][:, fIdx - 1, :, :, :].clone()
                flow = flows[fIdx - 1]
                fake_frame = self.resample(last_real_image, flow) * weight_mask + (
                    1 - weight_mask
                ) * fake_frame

            all_generated_frames[:, fIdx, :, :, :] = fake_frame

        self.all_gen_frames = all_generated_frames

    def multiscale_adversarial_loss(self, batch, for_discriminator=True):
        """Get the discriminator loss for multi-scale discriminator.

        If we are updating the discriminator, we are calculating the discriminator
        loss and vice-versa.
        """
        real_image = batch["image"][:, -1, :, :, :]

        if for_discriminator:
            # Fake; stop backprop to the generator by detaching fake_frame.
            self.generate_n_frames(batch)
            fake_frame = self.all_gen_frames[:, -1, :, :, :].detach()
            fake_concat = torch.cat((batch[self.hparams.encoder_input][:, -1, :, :, :], fake_frame), dim=1)
            real_concat = torch.cat((batch[self.hparams.encoder_input][:, -1, :, :, :], real_image), dim=1)

            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            predictions = self.multiscale_discriminator(fake_and_real)

            loss_D = 0.0
            for prediction in predictions:
                # Fake Detection and Real Detection Loss
                pred_fake, pred_real = torch.chunk(prediction, 2, dim=0)
                loss_D += self.criterion_GAN(pred_fake, False)
                loss_D += self.criterion_GAN(pred_real, True)

            return loss_D * 0.5
        else:
            fake_concat = torch.cat(
                (batch[self.hparams.encoder_input][:, -1, :, :, :], self.all_gen_frames[:, -1, :, :, :]), dim=1
            )
            predictions = self.multiscale_discriminator(fake_concat)
            loss_G = sum(self.criterion_GAN(pred, True) for pred in predictions) / len(predictions)

            return loss_G

    def temporal_adversarial_loss(self, batch, for_discriminator=True):
        """Temporal Discriminator Loss """

        if for_discriminator:
            self.generate_n_frames(batch)
            prev_frames_real = batch["image"][:, :-1, :, :, :]
            prev_frames_fake = self.all_gen_frames[:, :-1, :, :, :]
            encoder_input_real = batch[self.hparams.encoder_input][:, :-1, :, :, :]
            encoder_input_fake = batch[self.hparams.encoder_input][:, :-1, :, :, :]

            real_concat = torch.cat([encoder_input_real, prev_frames_real], dim=2)
            fake_concat = torch.cat([encoder_input_fake, prev_frames_fake], dim=2)

            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            predictions = self.temporal_discriminator(fake_and_real)

            loss_D = 0.0
            for prediction in predictions:
                # Fake Detection and Real Detection Loss
                pred_fake, pred_real = torch.chunk(prediction, 2, dim=0)
                loss_D += self.criterion_GAN(pred_fake, False)
                loss_D += self.criterion_GAN(pred_real, True)

            return loss_D * 0.5
        else:
            prev_frames_fake = self.all_gen_frames[:, :-1, :, :, :]
            encoder_input_fake = batch[self.hparams.encoder_input][:, :-1, :, :, :]
            fake_concat = torch.cat([encoder_input_fake, prev_frames_fake], dim=2)
            predictions = self.temporal_discriminator(fake_concat)
            loss_G = sum(self.criterion_GAN(pred, True) for pred in predictions) / len(predictions)

            return loss_G

    def get_prev_frames_and_maps(self, batch, fIdx, generated_frames):
        """Returns all previous frames, labels and the real images. """
        prev_n_frames_G = (
            generated_frames[:, fIdx - self.n_frames_now : fIdx, :, :, :].clone()
        )
        prev_n_labelmaps = {
            k: batch[k][:, fIdx - self.n_frames_now : fIdx, :, :, :].clone()
            for k in self.inputs
        }
        return prev_n_frames_G, prev_n_labelmaps

    def visualize(self, batch):
        grid = tensor_list_for_board(
            batch["image"], batch["cloth"], batch["densepose"], self.all_gen_frames
        )
        self.logger.experiment.add_image("current_tryon", grid, self.global_step)
