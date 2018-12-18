import nglview as nv

def _view3d(mol):
    """
    Draft of a function to visualize a molecule embedded in a notebook using nlgview
    """
    pdb_str = mol.dump_pdb('prot', to_file=False)
    view = nv.show_text(pdb_str)
    view.update_cartoon(color='#ed7300')
    return view

