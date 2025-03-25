import readers
import writers

print('------------------------------- CooReader -----------------------------')
cr = readers.CooReader('testdata/test.coo')
data = cr.load_data()
print('------------------------------- CsvWriter -----------------------------')
cw = writers.CsvWriter(filt=['id', 'east', 'north', 'elev'])    # write to stdout
cw.write_all(data)
print('------------------------------- CooWriter -----------------------------')
cw = writers.CooWriter()    # write to stdout
cw.write_all(data)
print('------------------------------- GeoReader -----------------------------')
cr = GeoReader('testdata/test.geo')
cr.load_data()
print('------------------------------- DmpWriter -----------------------------')
