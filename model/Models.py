faces_pixelate = "video:{'tag':'webcam'}-->Face_Trigger:{'prob':0.85}-->Blur_Face_Pixelate:{'blocks':5}"

faces_pixelate_with_resize = "video:{'tag':'webcam'}-->Max_Spec_Resize:{'max_width':640}-->Face_Trigger:{'prob':0.85}-->Blur_Face_Pixelate:{'blocks':5}\n" \
                             "video:{}-->Face_Trigger:{'prob':0.85}-->Blur_Face_Pixelate:{'blocks':5}"

nothing = "video:{'tag':'webcam'}"