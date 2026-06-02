# This file contains code for generating images with text written.
# Here we are usually trying to calcualte the similarity between two characters in terms of shapes.


from PIL import Image, ImageDraw, ImageFont

def generate_image(text, style="normal"):
    width = 280
    height = 420
    fontsize = 400
    h_shift = 0
    
    if style == "lower":
        h_shift = int(h_shift - fontsize/5)
    message = text
    img = Image.new('RGB', (width, height), color='white')
    imgDraw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arialn.ttf", fontsize)
    imgDraw.text((35, h_shift), message, fill=(0, 0, 0), font=font)
    # if style == "lower":
    #     img.crop((0, h_shift, width, height + h_shift))
    
    
    return img
    
if __name__ == '__main__':
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz!@#$%&(){[}]|?/><"
    lower_text = "gjpqy{[}]()|"
    half_text = "aceimnorsuvwxz"
    for char in letters:
        print("generating image for ", char)
        style = "lower" if char in lower_text else "half" if char in half_text else "normal"
        img = generate_image(char, style)
        path = "images/image_" + char + "_.png"
        img.save(path)
        print("saved image as ", path)
        