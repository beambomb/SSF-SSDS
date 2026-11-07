import nbformat as nbf

path = 'V57_End_to_End_Exorcism.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

old_line = "features = [c for c in train_full.columns if c not in drop_cols]"
new_line = "features = [c for c in train_full.columns if c not in drop_cols and train_full[c].dtype != 'object']"

found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and old_line in cell.source:
        cell.source = cell.source.replace(old_line, new_line)
        found = True
        break

if found:
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('Notebook fixed! object columns removed from features.')
else:
    print('Could not find the target line in the notebook.')
