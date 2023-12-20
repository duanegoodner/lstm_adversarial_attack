import toml

config = toml.load('test.toml')
full_name = config['full_name']  # This will be "John Doe"
print(full_name)