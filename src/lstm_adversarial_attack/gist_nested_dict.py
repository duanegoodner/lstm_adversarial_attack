import pprint

def my_function(dict_to_search, my_keys):
    result = dict_to_search
    result_dict = {}
    current_dict = result_dict

    try:
        for key in my_keys:
            current_dict[key] = {}
            current_dict = current_dict[key]
            result = result[key]

        current_dict[my_keys[-1]] = result  # Include the final key and value
        return result_dict
    except KeyError:
        # Handle the case where a key is not present in the dictionary
        return None

# Example usage:
my_dict = {
    "key_x": {
        "key_y": {
            "key_a": "desired_value_1",
            "key_b": "desired_value_2",
            "key_c": "desired_value_3"
        }
    }
}

keys_to_search_1 = ["key_x", "key_y", "key_a"]
result_dict_1 = my_function(my_dict, keys_to_search_1)

keys_to_search_2 = ["key_x", "key_y", "key_b"]
result_dict_2 = my_function(my_dict, keys_to_search_2)

pprint.pprint(result_dict_1)
pprint.pprint(result_dict_2)


