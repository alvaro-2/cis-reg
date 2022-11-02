"""
    Functions and classes to work with residue distances in proteins structures
"""

import csv
import operator
import re
from functools import reduce
from itertools import combinations, combinations_with_replacement, product
from typing import Dict, TextIO, Tuple

from Bio.PDB.Model import Model
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure

class Distances():
    '''
    Store and access distance data for residues from a protein structure.
    '''
    def __init__(self, dist_data):
        '''
        Creates a new instance from distance data.

        Distance data should be a list of tuples of five elements: (chain1, pos1, chain2, pos2, distance).
        :param dist_data: a list of (chain1, pos1, chain2, pos2, distance)
        '''
        dis = {}
        for ch1, po1, ch2, po2, dist in dist_data:
            if (ch1, po1) not in dis:
                dis[(ch1, po1)] = {}
            dis[(ch1, po1)][(ch2, po2)] = dist
        self._distances = dis

    def raw_distances(self) -> Tuple[str, int, str, int, float]:
        """
        Returns the distances data of the object as a list of tuples.
        Each tuples has five elements: (chain1, pos1, chain2, pos2, distance).
        """
        return [
            (chain1, pos1, chain2, pos2, dist)
            for (chain1, pos1), c_pos in self._distances.items()
            for (chain2, pos2), dist in c_pos.items()
        ]

    def of(self, chain_a, pos_a, chain_b, pos_b): #pylint: disable=invalid-name
        '''
        Retrieves distance for a residue pair.

        If the pair is not found, None is returned.
        :param chain_a: A string specifying the first residue chain.
        :param pos_a: An integer specifying the first residue position.
        :param chain_b: A string specifying the second residue chain.
        :param pos_b: An integer specifying the second residue position.
        '''
        pair1 = ((chain_a, pos_a))
        pair2 = ((chain_b, pos_b))
        if pair1 == pair2: # Special case for distance with the same residue.
            return 0
        distance = self._distances.get(pair1, {}).get(pair2)
        if not distance:
            distance = self._distances.get(pair2, {}).get(pair1)
        return distance

    def remap_positions(self, mapping):
        '''
        Remap index positions.

        If a positions could not be mapped it is excluded from the results.
        :param mapping: a dict that maps old positions to new positions.
        '''
        def _remap(dic):
            return {(chain, mapping[chain][pos]):value
                    for (chain, pos), value in dic.items()
                    if pos in mapping.get(chain, {})}

        self._distances = _remap({(c1, p1):_remap(r2)
                                  for (c1, p1), r2 in self._distances.items()})

    def is_contact(self, chain_a, pos_a, chain_b, pos_b, distance_cutoff=6.05): #pylint: disable=too-many-arguments
        '''
        Returns True if a given pair's distance is lower or equal than a given
        distance cutoff.
        :param chain_a: A string specifying the first residue chain.
        :param pos_a: An integer specifying the first residue position.
        :param chain_b: A string specifying the second residue chain.
        :param pos_b: An integer specifying the second residue position.
        :param distance_cutoff: a float with the distance cutoff (defaults to 6.05 angstroms)
        '''
        return self.of(chain_a, pos_a, chain_b, pos_b) <= distance_cutoff

    @staticmethod
    def _sum_true(boolean_list):
        return reduce(lambda a, b: a+(1 if b else 0), boolean_list, 0)

    def mean_intramolecular(self):
        """
        Return the mean number of intramolecular contacts across all residues for every chain.

            :param self: a Distances obj
        """
        def _pos_contacts(chain, pos1, all_positions):
            return [self.is_contact(chain, pos1, chain, pos2) for pos2 in all_positions
                    if not pos1 == pos2]
        all_residues = set(self._distances.keys()).union(
            {pair2 for pair1 in self._distances.keys() for pair2 in self._distances[pair1].keys()})
        all_chains = {chain for chain, pos in all_residues}
        pos_by_chain = {chain: [p for c, p in all_residues if c == chain] for chain in all_chains}

        n_contacts = {chain: [self._sum_true(_pos_contacts(chain, pos, pos_by_chain[chain]))
                              for pos in pos_by_chain[chain]]
                      for chain in all_chains}
        n_contacts = {chain: float(reduce(operator.add, n, 0)) / max(1, len(n)) for chain, n in n_contacts.items()}
        return n_contacts

    @staticmethod
    def from_contact_map(
            contact_map: Dict[Tuple[int, int], bool]) -> 'Distances':
        """
        Create a new Distance object from a contact map.
        Set contact to a distace of 1 and non contacts to 10.
        Sets the chain to be 'A'.
        """
        dist_data = []
        for (pos1, pos2), is_contact in contact_map.items():
            dist_data.append(
                ('A', pos1, 'A', pos2, 1 if is_contact else 10)
            )
        return Distances(dist_data)

def from_mitos(dist_file):
    '''
    Loads data of residue distances from a file generated by MIToS.

    Input data should look like:

    <pre>
    # model_i,chain_i,group_i,pdbe_i,number_i,name_i,model_j,chain_j,group_j,pdbe_j,number_j,name_j,distance
    1,A,ATOM,,55,LEU,1,A,ATOM,,56,LEU,1.3247309160731473
    </pre>

    :param dist_file: A string to a text file with the distance data.
    '''
    # model_i,chain_i,group_i,pdbe_i,number_i,name_i,model_j,chain_j,group_j,pdbe_j,number_j,name_j,distance
    # 1,A,ATOM,,55,LEU,1,A,ATOM,,56,LEU,1.3247309160731473
    #                       1    ,A  ,ATOM,  ,55   ,LEU  ,1   ,A  ,ATOM,  ,56   ,LEU ,1.3247309160731473
    d_pattern = re.compile(r"(\d+),(.),(.+),.*,(\d+),(.+),(\d+),(.),(.+),.*,(\d+),(.+),(.+)$")
    res = []
    with open(dist_file) as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("#"):
                match = re.match(d_pattern, line)
                try:
                    res.append((
                        match.group(2),      # Chain 1
                        int(match.group(4)), # Pos res 1
                        match.group(7),      # Chain 2
                        int(match.group(9)), # Pos res 2
                        float(match.group(11))))  # distance
                except (IndexError, AttributeError):
                    pass
        return res


def is_back_bone(atom):
    """
    Decides if an atom belongs to the backbone of a prototein by their name.
    """
    return atom.id == 'N' or atom.id == 'CA' or atom.id == 'CB'

def all_atoms_selector(atom1, atom2):
    """
    Accepts two any atoms.
    """
    #pylint: disable=unused-argument
    
    return True
def side_chain_selector(atom1, atom2):
    """
    Accepts two atoms that are part of the sidechain of an aminoacid.
    """
    return not is_back_bone(atom1) and not is_back_bone(atom2)
def carbon_alfa_selector(atom1, atom2):
    """
    Accepts two alpha carbon atoms
    """
    return atom1.id == 'CA' and atom2.id == 'CA'
def carbon_beta_selector(atom1, atom2):
    """
    Accepts two beta carbon atoms
    """
    return atom1.id == 'CB' and atom2.id == 'CB'

def _pick_pdb_model(pdb_source):
    if isinstance(pdb_source, Structure):
        struct = pdb_source
        model = list(struct.get_models())[0]
    elif isinstance(pdb_source, Model):
        model = pdb_source
    elif isinstance(pdb_source, str):
        parser = PDBParser()
        struct = parser.get_structure('XXXX', pdb_source)
        model = list(struct.get_models())[0]
    return model

def _shorter_distance_between_residues(
        res1, res2, chain1, chain2, atom_selector):
    min_dist = float('inf')
    min_res_data = None
    for atom1, atom2 in product(res1, res2):
        if not atom1.id.startswith('H') and not atom2.id.startswith('H') and atom_selector(atom1, atom2):
            dist = atom1-atom2
            if dist < min_dist:
                min_dist = dist
                sorted_pair = sorted(
                    [(chain1.id, res1.id[1], res1.resname, atom1.id),
                     (chain2.id, res2.id[1], res2.resname, atom2.id)])
                min_res_data = [item for res in sorted_pair
                                for item in res] + [dist]
    return min_res_data

def calculate_distances(
        pdb_source,
        atom_selector=carbon_beta_selector,
        include_extra_info=False):
    """
    Compute distances between residues
        :param pdb_source: a path to a pdb file, a Bio.PDB.Structure or a
            Bio.PDB.Model
        :param atom_selector=all_atoms_selector: a function that allows to
            select pairs of atoms to include into the distance calculation.
        :param include_extra_info=False: If True adds residue name and atom name
            for each contacting atom to the output.
    """
    model = _pick_pdb_model(pdb_source)
    chains = model.get_chains()
    out = []
    for chain1, chain2 in combinations_with_replacement(chains, 2):
        if chain1 is chain2:
            res_iter = combinations(chain1, 2)
        else:
            res_iter = product(chain1, chain2)
        for res1, res2 in res_iter:
            if not res1 is res2:
                min_res_data = _shorter_distance_between_residues(
                    res1, res2, chain1, chain2, atom_selector)
            if min_res_data:
                if include_extra_info:
                    out.append(min_res_data)
                else:
                    out.append([
                        min_res_data[0],
                        min_res_data[1],
                        min_res_data[4],
                        min_res_data[5],
                        min_res_data[8]
                    ])
    return out

def save_distances(dist_data, outfile):
    """
    Saves distance data to a file.

    Despite the content of the dist_data list, the output file will contain
    nine fields. Missing data fill filled with NA fields.
        :param dist_data: data generated with calculate_distance function
        :param outfile: exported file
    """
    with open(outfile, 'w') as text_handle:
        for row in dist_data:
            if len(row) == 9: # Data with additional info.
                pass
            elif len(row) == 5: # Data with no additional info.
                row = [
                    str(row[0]),
                    str(row[1]),
                    "NA",
                    "NA",
                    str(row[2]),
                    str(row[3]),
                    "NA",
                    "NA",
                    str(row[4])
                ]
            else:
                raise ValueError("Distance data has wrong number of element")
            text_handle.write(" ".join([str(x) for x in row]))
            text_handle.write("\n")

def read_distances(ditance_file, add_extra_info=False):
    """
    Read distance data file.
    """
    out = []
    with open(ditance_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            if len(row) == 9: # Data with additional info.
                if add_extra_info:
                    out.append([
                        row[0],
                        int(row[1]),
                        row[2],
                        row[3],
                        row[4],
                        int(row[5]),
                        row[6],
                        row[7],
                        float(row[8])
                    ])
                else:
                    out.append([
                        row[0],
                        int(row[1]),
                        row[4],
                        int(row[5]),
                        float(row[8])
                    ])
        return out

def contact_map_from_scpe(
        file_handle: TextIO,
        quaternary: bool = False,
        chars: str = "10") -> Dict[Tuple[int, int], bool]:
    """
    Read contact from SCPE output.

    The file content should have as many lines as positions in the protein.
    Each line should have characters from chars argument, separated by a space.
    There should be as many characters in every line as positions in the
    protein.

    Returns a dict object from position pairs to boolean values that indicates
    that the corresponding pair is in contact or not. Position index start at 1.

        :param file_handle: handle to the contact map file.
        :param quaternary: a boolean value that indicates if quaternary contacts
            should be included.
        :param chars="10": Characters accepted in the contact map.
            This argument is expected to have a length of two characters.
            The first is the value for residue pairs in contact,
            the second one is the value for non contacts.
    """
    contact_line_pattern = re.compile(f"[{chars}]( [{chars}])+$")
    qtag = "quaternary"
    ttag = "terciary"
    tags = [qtag, ttag]
    target_tag = qtag if quaternary else ttag
    correct_section = False
    position_index = 0
    contact_map = {}
    for line in file_handle:
        line = line.lower()
        c_match = re.match(contact_line_pattern, line)
        if c_match and correct_section:
            position_index += 1
            c_contacts = line.split()
            contact_map.update({
                (position_index, x+1): c == chars[0]
                for x, c in enumerate(c_contacts)})
        else:
            line = line.strip()
            if any([line == t for t in tags]):
                correct_section = line == target_tag
    return contact_map

def contact_map_from_text(
        file_handle: TextIO,
        chars: str = "10") -> Dict[Tuple[int, int], bool]:
    """
    Reads the content of a file object as a contact map.

    The file content should have as many lines as positions in the protein.
    Each line should have characters from chars argument, separated by a space.
    There should be as many characters in every line as positions in the
    protein.

    Returns a dict object from position pairs to boolean values that indicates
    that the corresponding pair is in contact or not. Position index start at 1.

        :param file_handle: handle to the contact map file.
        :param chars="10": Characters accepted in the contact map.
             This argument is expected to have a length of two characters.
             The first is the value for residue pairs in contact,
             the second one is the value for non contacts.
    """
    contact_line_pattern = re.compile(f"[{chars}]( [{chars}])+$")
    position_index = 0
    contact_map = {}
    for line in file_handle:
        line = line.strip().lower()
        c_match = re.match(contact_line_pattern, line)
        if c_match:
            position_index += 1
            c_contacts = line.split()
            contact_map.update({
                (position_index, x+1): c == chars[0]
                for x, c in enumerate(c_contacts)})
    return contact_map
