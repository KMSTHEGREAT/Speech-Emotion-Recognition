import ser

audio = input('Enter audio file name: ')
out = ser.Predict(audio)
print(out)