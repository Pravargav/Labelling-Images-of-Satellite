from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

class MatrixOperations:
    @staticmethod
    def add_matrices(matrix_a, matrix_b):
        return tf.add(matrix_a, matrix_b).numpy()

    @staticmethod
    def multiply_matrices(matrix_a, matrix_b):
        return tf.matmul(matrix_a, matrix_b).numpy()

    @staticmethod
    def elementwise_multiply(matrix_a, matrix_b):
        return tf.multiply(matrix_a, matrix_b).numpy()

    @staticmethod
    def transpose_matrix(matrix):
        return tf.transpose(matrix).numpy()

    @staticmethod
    def inverse_matrix(matrix):
        return tf.linalg.inv(matrix).numpy()

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('\n')
    return np.array([list(map(float, row.split())) for row in rows])

@app.route('/home')
def index2():
    return render_template('index.html')

@app.route('/')
def index():
    return "<p>Hello, World!</p>"

@app.route('/matrix_operations', methods=['POST'])
def matrix_operations():
    matrix_a = parse_matrix(request.form['matrix_a'])
    matrix_b = parse_matrix(request.form['matrix_b'])

    result_addition = MatrixOperations.add_matrices(matrix_a, matrix_b)
    result_multiplication = MatrixOperations.multiply_matrices(matrix_a, matrix_b)
    result_elementwise_multiply = MatrixOperations.elementwise_multiply(matrix_a, matrix_b)
    result_transpose_a = MatrixOperations.transpose_matrix(matrix_a)
    result_transpose_b = MatrixOperations.transpose_matrix(matrix_b)

    try:
        result_inverse_a = MatrixOperations.inverse_matrix(matrix_a)
        result_inverse_b = MatrixOperations.inverse_matrix(matrix_b)
    except tf.errors.InvalidArgumentError:
        result_inverse_a = "Matrix A is not invertible"
        result_inverse_b = "Matrix B is not invertible"

    return render_template('result.html',
                           matrix_a=matrix_a,
                           matrix_b=matrix_b,
                           result_addition=result_addition,
                           result_multiplication=result_multiplication,
                           result_elementwise_multiply=result_elementwise_multiply,
                           result_transpose_a=result_transpose_a,
                           result_transpose_b=result_transpose_b,
                           result_inverse_a=result_inverse_a,
                           result_inverse_b=result_inverse_b)



if __name__ == '__main__':
    app.run(debug=True)
