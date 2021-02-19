from django import forms
from .models import clip

class ClipPostForm(forms.ModelForm):
	class Meta:
		model = clip
		fields = ['contents'] # '__all__' 