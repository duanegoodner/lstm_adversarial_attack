def replace_inf(nested_list):
    if isinstance(nested_list, list):
        return [replace_inf(item) for item in nested_list]
    elif nested_list == "inf":
        return float("inf")
    else:
        return nested_list

# Example usage:
nested_list_2d = [[1, "inf"], [3, 4]]
nested_list_3d = [[[1, 2], ["inf", 4]], [[5, "inf"], [7, 8]]]

print(replace_inf(nested_list_2d))
print(replace_inf(nested_list_3d))
