import numpy as np
import matplotlib.pyplot as plt

# Use NumPy's linalg.solve function to check if vector  𝑡⃗   is within the span of the other two vectors,  𝑣⃗   and  𝑤⃗ .
def check_vector_span(set_of_vectors, vector_to_check):
    # Creates an empty vector of correct size
    vector_of_scalars = np.asarray([None]*set_of_vectors.shape[0])

    # Solves for the scalars that make the equation true if vector is within the span
    try:
        # TODO: Use np.linalg.solve() function here to solve for vector_of_scalars
        vector_of_scalars = np.linalg.solve(set_of_vectors, vector_to_check)
        if not (vector_of_scalars is None):
            print("\nVector is within span.\nScalars in s:", vector_of_scalars)
    # Handles the cases when the vector is NOT within the span
    except Exception as exception_type:
        if str(exception_type)=='Singular matrix':
            print('\nNo single solution\nVector is NOT within span')
        else:
            print('\nUnexpected Exception Error: ', exception_type)
    return vector_of_scalars

# Creates matrix t (right side of the augmented matrix).
t = np.array([4, 11])
# Creates matrix vw (left side of the augmented matrix).
vw = np.array([[1, 2], [3, 5]])
# Prints vw and t
print('\nCase 1:\nMatrix vw:', vw, '\nVector t:', t, sep='\n')
check_vector_span(vw, t)

plt.plot([4/1,0],[0,4/2],'b',linewidth=3)   # v line
plt.plot([11/3,0],[0,11/5],'c-.',linewidth=3)   # w line
plt.plot([2],[1],'ro',linewidth=3)  # t dot
plt.xlabel('Single Solution')
plt.show()

# Call to check a new set of vectors vw2 and t2
vw2 = np.array([[1, 2], [2, 4]])
t2 = np.array([6, 12])
print("\nCase 2:\nNew Vectors:\n Matrix vw2:", vw2, "\nVector t2:", t2, sep="\n")
# Call to check_vector_span
s2 = check_vector_span(vw2,t2)

plt.plot([6/1,0],[0,6/2],'b',linewidth=3)   # v line
plt.plot([12/2,0],[0,12/4],'c-.',linewidth=3)   # w line
plt.xlabel('Redundant Equations')
plt.show()

# Call to check a new set of vectors vw3 and t3
vw3 = np.array([[1, 2], [1, 2]])
t3 = np.array([6, 10])
print("\nCase 3:\nNew Vectors:\n Matrix vw3:", vw3, "\nVector t3:", t3, sep="\n")
# Call to check_vector_span
s3 = check_vector_span(vw3,t3)

plt.plot([6/1,0],[0,6/2],'b',linewidth=3)   # v line
plt.plot([10/1,0],[0,10/2],'c-.',linewidth=3)   # w line
plt.xlabel('No Solution')
plt.show()

