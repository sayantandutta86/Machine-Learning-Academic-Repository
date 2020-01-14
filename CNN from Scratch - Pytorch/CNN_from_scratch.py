'''Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). You should write your own code for convolutions (e.g., do not use SciPy's convolution function). The convolution network should have a single hidden layer with multiple channels. It should achieve at least 94% accuracy on the Test Set. '''
#Sayantan Dutta

import h5py
import numpy as np

#Convolution Class
class ConvClass_3x3:

  def __init__(self, filter_count):

    self.filter_count = filter_count

    self.filter_array = np.random.randn(filter_count, 3, 3) / 9

  def generate_conv_area(self, image):

    height, width = image.shape

    for i in range(height - 2):

      for j in range(width - 2):

        conv_area = image[i:(i + 3), j:(j + 3)]

        yield conv_area, i, j

  def forwardpass(self, input_image):

    self.previous_input_image = input_image

    height, width = input_image.shape

    output_conv_image = np.zeros((height - 2, width - 2, self.filter_count))

    for conv_area, i, j in self.generate_conv_area(input_image):

      output_conv_image[i, j] = np.sum(conv_area * self.filter_array, axis=(1, 2))

    return output_conv_image

  def backwardpass(self, output_gradient, lr):

    gradient = np.zeros(self.filter_array.shape)

    for conv_area, i, j in self.generate_conv_area(self.previous_input_image):
      for f in range(self.filter_count):
        gradient[f] += output_gradient[i, j, f] * conv_area

    self.filter_array -= lr * gradient


    return None

#Maxpooling Class
class MaxPooling_2x2:


  def generate_conv_area(self, image):

    height, width, _ = image.shape

    height_new = height // 2

    width_new = width // 2

    for i in range(height_new):

      for j in range(width_new):

        conv_area = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]

        yield conv_area, i, j

  def forwardpass(self, input_image):

    self.previous_input_image = input_image

    height, weight, filter_count = input_image.shape

    output_conv_image = np.zeros((height // 2, weight // 2, filter_count))

    for conv_area, i, j in self.generate_conv_area(input_image):

      output_conv_image[i, j] = np.amax(conv_area, axis=(0, 1))

    return output_conv_image

  def backwardpass(self, output_gradient):

    gradient = np.zeros(self.previous_input_image.shape)

    for conv_area, i, j in self.generate_conv_area(self.previous_input_image):

      height, width, f_count = conv_area.shape

      filter_max_value = np.amax(conv_area, axis=(0, 1))

      for h in range(height):

        for w in range(width):

          for f in range(f_count):

            if conv_area[h, w, f] == filter_max_value[f]:

              gradient[i * 2 + h, j * 2 + w, f] = output_gradient[i, j, f]

    return gradient

#Fully Conencted Layer
class FC_layer:

  def __init__(self, input_len, nodes):

    self.weights = np.random.randn(input_len, nodes) / input_len

    self.biases = np.zeros(nodes)

  def forwardpass(self, input_image):

    self.previous_input_shape = input_image.shape

    input_image = input_image.flatten()

    self.previous_input_image = input_image

    input_units, output_units = self.weights.shape

    self.totals = np.dot(input_image, self.weights) + self.biases

    totals_relu = np.maximum(self.totals, 0)

    self.previous_lf1 = totals_relu

    exp = np.exp(totals_relu)

    return exp / np.sum(exp, axis=0)

  def backwardpass(self, output_gradient, lr):

    for i, out_grad in enumerate(output_gradient):

      if out_grad == 0:

        continue

      lf1_exp = np.exp(self.previous_lf1)

      sum_lf1_exp = np.sum(lf1_exp)

      grad_out_lf1 = -lf1_exp[i] * lf1_exp  / (sum_lf1_exp ** 2)

      grad_out_lf1[i] = lf1_exp[i] * (sum_lf1_exp - lf1_exp[i])  / (sum_lf1_exp ** 2)

      grad_lf1_w1 = self.previous_input_image

      grad_lf1_b1 = 1

      grad_lf1_input = self.weights

      grad_loss_lf1 = out_grad * grad_out_lf1

      grad_loss_w1 = grad_lf1_w1[np.newaxis].T @ grad_loss_lf1[np.newaxis]
      grad_loss_b1 = grad_loss_lf1 * grad_lf1_b1
      grad_loss_input = grad_lf1_input @ grad_loss_lf1

      self.weights -= lr * grad_loss_w1
      self.biases -= lr * grad_loss_b1

      return grad_loss_input.reshape(self.previous_input_shape)


#Load dataset
data = h5py.File('MNISTdata.hdf5', 'r')

#Load train and test data

X_train = np.float32(data['x_train']).reshape(-1, 28, 28)

X_test = np.float32(data['x_test']).reshape(-1, 28, 28)

y_train = np.int32(np.array(data['y_train'])).reshape(-1, 1)

y_test = np.int32(np.array(data['y_test'])).reshape(-1, 1)

data.close()

train_records = y_train.shape[0]

index_shuffled = np.random.permutation(train_records)

X_train, y_train = X_train[index_shuffled][:][:], y_train[index_shuffled][:][:]

filter_count = 32

conv = ConvClass_3x3(filter_count)

maxpool = MaxPooling_2x2()

fc_layer = FC_layer(13 * 13 * filter_count, 10)

def forwardpass_main(input_image, y_true):

  y_pred = conv.forwardpass((input_image))

  y_pred = maxpool.forwardpass(y_pred)

  y_pred = fc_layer.forwardpass(y_pred)

  loss = -np.log(y_pred[y_true])

  correct = 1 if np.argmax(y_pred) == y_true else 0

  return y_pred, loss, correct


def training(input_image, y_true, lr):

  y_pred, loss, correct = forwardpass_main(input_image, y_true)

  grad = np.zeros(10)

  grad[y_true] = -1 / y_pred[y_true]

  grad = fc_layer.backwardpass(grad, lr)

  grad = maxpool.backwardpass(grad)

  grad = conv.backwardpass(grad, lr)

  return loss, correct


for epoch in range(10):
  print(f'Epoch: {epoch + 1}')

  # Shuffled training data
  index_shuffled = np.random.permutation(len(X_train))

  X_train, y_train = X_train[index_shuffled][:][:], y_train[index_shuffled][:][:]

  loss = 0
  accuracy = 0
  step = 500

  for i, (image, label) in enumerate(zip(X_train[:10000], y_train[:10000])):
    if i % step == step-1:
      print(f'[Step {i+1}] of Epoch {epoch}, for last {step} steps: Average Loss {loss/step} | Accuracy: {100*(accuracy/step)}%')

      loss = 0
      accuracy = 0
    #learning rate
    if i<=2000:
      lr=0.005
    elif (i>2000 and i<=7000):
      lr=0.001
    elif i>7000:
      lr=0.0005

    l, correct = training(image, label, lr)
    loss += l
    accuracy += correct


  print(f'\n Test results of the CNN after epoch {epoch+1}')
  test_loss = 0
  test_num_correct = 0
  for image, label in zip(X_test, y_test):
    _, l, correct = forwardpass_main(image, label)
    test_loss += l
    test_num_correct += correct

  num_tests = len(X_test)
  print('Test Loss:', test_loss / num_tests)
  print('Test Accuracy:', test_num_correct / num_tests)