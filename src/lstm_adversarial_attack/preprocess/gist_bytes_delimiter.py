def read_bytes_until_delimiter(data, delimiter):
    result = b""  # Initialize an empty bytes object to store the read data
    index = data.find(delimiter)  # Find the index of the delimiter in the bytes object

    if index != -1:
        result = data[:index + len(delimiter)]  # Include the delimiter in the result
        remaining_data = data[index + len(delimiter):]  # Store the remaining data
    else:
        result = data  # If delimiter not found, read the entire bytes object
        remaining_data = b""

    return result, remaining_data

# Example usage
data = b"Hello|World|This|Is|Some|Data"
delimiter = b"|"
my_result, my_remaining_data = read_bytes_until_delimiter(data, delimiter)

print("Result:", my_result)
print("Remaining Data:", my_remaining_data)
