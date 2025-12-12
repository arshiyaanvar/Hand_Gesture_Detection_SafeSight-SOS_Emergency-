from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "dataset_split/train"

gen = ImageDataGenerator(rescale=1/255)
flow = gen.flow_from_directory(train_dir)

print(flow.class_indices)
