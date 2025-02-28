import pandas as pd

# Create DataFrame
data = {
    "ID": list(range(19)),
    "Measurement": ["potassium", "calcium", "ph", "pco2", "lactate", "albumin",
                    "bun", "creatinine", "sodium", "bicarbonate", "platelet",
                    "glucose", "magnesium", "heartrate", "sysbp", "diasbp",
                    "tempc", "resprate", "spo2"]
}
df = pd.DataFrame(data)

# Apply styling
styled = df.style.set_table_styles(
    [{'selector': 'th', 'props': [('background', 'lightgray'), ('font-weight', 'bold')]}]
).set_properties(**{'text-align': 'left'})

# Save as HTML file
html_path = "table_output.html"
styled.to_html(html_path)

print(f"Table saved! Open '{html_path}' in a browser.")
