import pandas as pd

df = pd.read_excel("/mnt/data/Timeline + BOM.xlsx")

html_rows = ""

for i, row in df.iloc[1:].iterrows():
    item = row["Unnamed: 0"]
    link = row["Unnamed: 1"]
    source = row["Unnamed: 2"]
    unit = float(row["Unnamed: 3"])
    qty = float(row["Unnamed: 4"])
    total = unit * qty

    html_rows += (
        f"<tr>\n"
        f'    <td><a href="{link}">{item}</a></td>\n'
        f"    <td>{source}</td>\n"
        f"    <td>{int(qty)}</td>\n"
        f"    <td>${unit:.2f}</td>\n"
        f"    <td>${total:.2f}</td>\n"
        f"</tr>\n\n"
    )

with open("output.txt", "w") as file:
    file.write(html_rows + "\n")
