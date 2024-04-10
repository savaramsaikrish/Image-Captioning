import models as models
import preprocessingmain as pp

obj= pp.preprocess_data()

#directory path
dir_path= "foodmain"

#visualise images
obj.visualize_images(dir_path,nimages=5)

#Preprocess data
eye_df,train,labels=obj.preprocess(dir_path)
print(train)
print(labels)

# strain and label datframe to csv
eye_df.to_csv("Image_df.csv",index=False)

#0train genarator, test genarator , validate genarator
tr_gen,tt_gen,va_gen=obj.generate_train_test_split(eye_df,train,labels)

ms= models.DeepANN()

input_shape=(128,128,3)
mss=[]

#model2
#mss.append(ms.simple_ANN(input_shape=input_shape,optimizer="adam"))
#mss.append(ms.simple_ANN(input_shape=input_shape,optimizer="sgd"))
#mss.append(ms.simple_ANN(input_shape=input_shape,optimizer="rmsprop"))

#model6
#mss.append(ms.cnn_model(input_shape=input_shape,optimizer="adam"))
#mss.append(ms.cnn_model(input_shape=input_shape,optimizer="sgd"))
#mss.append(ms.cnn_model(input_shape=input_shape,optimizer="rmsprop"))

#model3
#mss.append(ms.simple_model_multiclass(input_shape=input_shape,optimizer="sgd"))

#model7
#mss.append((ms.cnn_add_regularize()))

#model8
#mss.append(ms.cnn_vgg())

#model9
#mss.append(ms.create_rnn_model(input_shape=input_shape,no_of_classes=3))

#model10
mss.append(ms.create_LSTM_rnn_model(input_shape=input_shape,no_of_classes=3))

#model1
#mss.append(ms.simple_model(input_shape=input_shape,op="adam"))

#compare
models.compare_model(mss, tr_gen, va_gen, tt_gen, epochs=3)