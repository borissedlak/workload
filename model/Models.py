faces_pixelate = "video:{'tag':'webcam'}-->Max_Spec_Resize:{'max_width':640}-->Face_Trigger:{'prob':0.85}-->Age_Trigger:{'prob': 0.85, 'label': '(25-32)', 'debug': True}-->Blur_Face_Pixelate:{'blocks':5}"

faces_pixelate_with_resize = "video:{'tag':'webcam'}-->Max_Spec_Resize:{'max_width':640}-->Face_Trigger:{'prob':0.85}-->Blur_Face_Pixelate:{'blocks':5}\n" \
                             "video:{}-->Face_Trigger:{'prob':0.85}-->Blur_Face_Pixelate:{'blocks':5}"

nothing = "video:{'tag':'webcam'}"
