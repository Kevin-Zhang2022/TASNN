import struct
def Fun_F2B(Inp):
    return bin(struct.unpack('!I', struct.pack('!f', Inp))[0])[2:].zfill(32)
