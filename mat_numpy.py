# nはnumpy.ndarray
def mul(n):
    print("from python")
    print("received: " + str(n))
    print("end from python")

    # ndarrayのタプルを返す
    return (2 * n, 3 * n)
