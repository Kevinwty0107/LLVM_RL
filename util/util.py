def flatten(xss):
    ret = []
    for xs in xss:
        for x in xs:
            ret.append(x)
    return ret


def denoise(x, denoise_thres, strength=3):
    y = x
    if abs(y) > denoise_thres:
        return y
    else:
        return (y / denoise_thres) ** strength * denoise_thres