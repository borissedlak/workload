faces_pixelate = "video:{'tag':'webcam'}-->Face_Trigger:{'prob':0.85}-->Blur_Area_Pixelate:{'blocks':5}"

audio_replace = "audio:{'tag':'microphone'}"

faces_pixelate_with_resize = "video:{'tag':'webcam'}-->Max_Spec_Resize:{'max_width':640}-->Face_Trigger:{'prob':0.85}-->Blur_Area_Pixelate:{'blocks':5}\n" \
                             "video:{}-->Face_Trigger:{'prob':0.85}-->Blur_Area_Pixelate:{'blocks':5}"

nothing = "video:{'tag':'webcam'}"

model_1 = "video:{'tag':'webcam'}-->Face_Trigger:{'prob':0.85}-->Blur_Area_Pixelate:{'blocks':5}"
model_2 = "video:{'tag':'webcam'}-->Face_Trigger:{'prob':0.85}-->Age_Trigger:{'prob':0.85, 'label':'(25-32)'}-->Blur_Area_Pixelate:{'blocks':5}"
model_3 = "video:{'tag':'webcam'}-->Face_Trigger:{'prob':0.85}-->Gender_Trigger:{'prob':0.85, 'label':'Male'}-->Blur_Area_Pixelate:{'blocks':5}"
model_4 = "video:{'tag':'webcam'}-->Face_Trigger:{'prob':0.85}-->Fill_Area_Box:{}"

