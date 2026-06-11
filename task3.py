# Matrix Operations Tool
# Internship Task
# Using Python and NumPy

import numpy as np


def input_matrix(name):
    print(f"\nEnter values for {name}")
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))

    print("Enter matrix values row-wise separated by spaces:")

    matrix = []

    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))

        if len(row) != cols:
            print("Invalid number of values!")
            return input_matrix(name)

        matrix.append(row)

    return np.array(matrix)


def display_matrix(title, matrix):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    print(matrix)
    print("=" * 40)


def matrix_addition(A, B):
    try:
        return A + B
    except ValueError:
        return "Addition not possible (Different dimensions)"


def matrix_subtraction(A, B):
    try:
        return A - B
    except ValueError:
        return "Subtraction not possible (Different dimensions)"


def matrix_multiplication(A, B):
    try:
        return np.matmul(A, B)
    except ValueError:
        return "Multiplication not possible (Invalid dimensions)"


def matrix_transpose(A):
    return A.T


def matrix_determinant(A):

    if A.shape[0] != A.shape[1]:
        return "Determinant only exists for square matrices"

    return np.linalg.det(A)



def menu():

    print("\n******** MATRIX OPERATIONS TOOL ********")

    print("""
1. Addition
2. Subtraction
3. Multiplication
4. Transpose
5. Determinant
6. Exit
""")


# Main Program

while True:

    menu()

    choice = input("Choose an operation (1-6): ")


    if choice == "1":

        A = input_matrix("Matrix A")
        B = input_matrix("Matrix B")

        display_matrix(
            "Matrix A",
            A
        )

        display_matrix(
            "Matrix B",
            B
        )

        result = matrix_addition(A, B)

        display_matrix(
            "Addition Result",
            result
        )


    elif choice == "2":

        A = input_matrix("Matrix A")
        B = input_matrix("Matrix B")

        result = matrix_subtraction(A, B)

        display_matrix(
            "Subtraction Result",
            result
        )


    elif choice == "3":

        A = input_matrix("Matrix A")
        B = input_matrix("Matrix B")

        result = matrix_multiplication(A, B)

        display_matrix(
            "Multiplication Result",
            result
        )


    elif choice == "4":

        A = input_matrix("Matrix")

        result = matrix_transpose(A)

        display_matrix(
            "Transpose Result",
            result
        )


    elif choice == "5":

        A = input_matrix("Matrix")

        result = matrix_determinant(A)

        display_matrix(
            "Determinant Result",
            result
        )


    elif choice == "6":

        print("\nThank you for using Matrix Operations Tool!")
        break


    else:

        print("Invalid choice! Please select between 1-6.")