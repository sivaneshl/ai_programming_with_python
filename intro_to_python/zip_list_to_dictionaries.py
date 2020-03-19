# Use zip to create a dictionary cast that uses names as keys and heights as values.

cast_names = ["Barney", "Robin", "Ted", "Lily", "Marshall"]
cast_heights = [72, 68, 72, 66, 76]

# cast = {key: value for key, value in zip(cast_names, cast_heights)}
cast = dict(zip(cast_names, cast_heights))
print(cast)