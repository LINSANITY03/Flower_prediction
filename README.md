# Prediction model for classifying flower

In this project we use pandas to collect data from a source and in-built tensorflow model to train the data.

To run this project,

- Create a virutal environment and activate the environment.
  `  virtualenv venv
\venv\Scripts\activate`

- Install the required dependencies.
  `pip install -r requirements.txt
`

- Run the **main.py** file.
  `python main.py`

**1. Data Collection:**
We get flower training and evaluation data from google drive links.

```
import tensorflow as tf
...

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

```

**2. Feature Extraction:**
Using the in-built feature column function of tensorflow, we get all the unique value from each column of the tf file.

```
# Feature columns describe how to use the input.
# We use the inbuilt function in tensorflow to get all the unique value represented in the data of certain features
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

**3. Data Preparation:**
We need to make sure the data are in appropritate format for the tensorflow model. So, we convert the datas into data.Dataset object using tf.data.Dataset function

```
# convert the data into Dataset format
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)
```

**4. Choosing a Model:**
Our goal is to predict the flower based on given length. So, a DNN classifier model would do the trick.

```
# Creating a model
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)
```

**5. Training the model:**
We use the data we convert to data.Dataset object to the model.

```
# training the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# We include a lambda to avoid creating an inner function previously
```

**6. Evaluate the model:**
Test the unseen dataset to measure the performance of the trained model.

```
# Evaluate the model accuracy by running the same data
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
'''Test set accuracy: 0.900'''
```

**7. Make prediction:**
Using the evaluated model predict the flower species and return flower name with probability for better readability.

```
predictions = classifier.predict(input_fn=lambda: user_input(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

'''
Please type numeric values as prompted.
SepalLength: 5.1
SepalWidth: 5.9
PetalLength: 1.7
PetalWidth: 0.5
...
Prediction is "Setosa" (97.4%)
'''
```
