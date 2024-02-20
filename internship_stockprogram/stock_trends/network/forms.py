from django import forms
from .models import Closet


class ImageUploadForm(forms.ModelForm):
    cloth_size = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control', 'list':'', 'required':'required', 'placeholder':'Minimum Bid'}), required=False)
    cloth_item = forms.CharField(widget=forms.TextInput(attrs={'type':'text','class':'form-control', 'aria-describedby':'inputGroup-sizing-lg', 'required':'required'}))
    image = forms.ImageField(widget=forms.ClearableFileInput(attrs={'class': 'form-control', 'id':'inputGroupFile02', 'required':'required'}))
    cloth_category = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'cols': 40, 'class': 'form-control', 'aria_label': 'Description', 'required':'required'}))
    cloth_color = forms.CharField(widget=forms.Textarea(attrs={'rows': 5, 'cols': 40, 'class': 'form-control', 'aria_label': 'Description', 'required':'required'}))

    class Meta:
        model = Closet
        fields = ['cloth_size', 'user', 'cloth_item', 'cloth_color', 'cloth_category', 'image']