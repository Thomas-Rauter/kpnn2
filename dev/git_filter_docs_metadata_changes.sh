pip install nbstripout
nbstripout --install
git config filter.nbstripout.clean "$(git config filter.nbstripout.clean) --keep-output --keep-id"
git config diff.ipynb.textconv "$(git config diff.ipynb.textconv) --keep-output --keep-id"
nbstripout --status
