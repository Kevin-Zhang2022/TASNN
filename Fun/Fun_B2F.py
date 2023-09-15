import struct
def Fun_B2F(Inp):
    return struct.unpack('!f', struct.pack('!I', int(Inp, 2)))[0]
