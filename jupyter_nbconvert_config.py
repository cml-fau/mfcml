# Picked up automatically by `jupyter nbconvert` when run from the repo root.
# Cells are hidden in the exported HTML via their tags (Jupyter: Property
# Inspector -> Cell Tags; VS Code: cell "..." menu -> Add Cell Tag):
#   remove-cell    drop the cell entirely (code + output)
#   remove-input   keep the output, hide the source
#   remove-output  keep the source, hide the output
c = get_config()  # noqa: F821

c.TagRemovePreprocessor.enabled = True
c.TagRemovePreprocessor.remove_cell_tags = ("remove-cell",)
c.TagRemovePreprocessor.remove_input_tags = ("remove-input",)
c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove-output",)
