import model

m = model.Model('TrainedSet5.h5')
m.LoadTrainingData("D:\\TrainingData\\Set6","Set6")
m.CompileAndFit(fileName='TrainedSet6')

print("Done!")



#m.GrayItAll("D:\\TrainingData\\Set6","Set6")