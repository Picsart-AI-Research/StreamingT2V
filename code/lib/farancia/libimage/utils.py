from IPython.display import Image as IpyImage

def bytes2html(data, width='auto'):
    img_obj = IpyImage(data=data, format='JPG')
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'<img src="data:{mimetype};base64,{b64value}" style="width: {width}; max-width: 100%">'