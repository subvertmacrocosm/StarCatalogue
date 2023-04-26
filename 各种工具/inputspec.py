spec1 = fits.open(root + '\\' + name)
spec = spec1[0].data[0]
spec = spec.byteswap().newbyteorder()
wave = spec1[0].data[0]
wave_min_st = np.where(spec1[0].data[2] == min_wave[0])
wave_max_st = np.where(spec1[0].data[2] == max_wave[0])
wave = wave[wave_min_st, wave_max_st]
print('{},,,{}'.format(wave.min(), wave.max()))
spec = mms(spec, spec.min(), spec.max())
spec = np.maximum(spec, 0)