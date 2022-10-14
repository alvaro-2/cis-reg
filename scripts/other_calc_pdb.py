import warnings
import pandas as pd
import os
import Bio
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from io import StringIO
from functools import reduce
from scipy.spatial import distance
from operator import add
from math import sqrt, ceil
from itertools import combinations
from Bio.PDB.Superimposer import Superimposer
import numpy as np
from src.utilities import stat_data

import tarfile
import gc
import shutil

class OTHER_calc:
    def __init__(self, path_maxit, pdb_content, path_folder, folder_data, ensembl_id, folder_pdb, folder_cif, path_tmp): 
         
        self.__files = {}
        
        warnings.filterwarnings("ignore") 
        parser = PDBParser()
        pdb_structure = parser.get_structure(id="XXX1", file=pdb_content)
        string_io = StringIO()
        pdbio = PDBIO()
        pdbio.set_structure(pdb_structure)     
        pdbio.save(string_io, select=SelectCA())
        string_io = StringIO(string_io.getvalue())
        pdb_structure_onlyCA = parser.get_structure("XXX1", string_io)
        
        self.__radius_and_dmax(path_folder, folder_data, ensembl_id, pdb_structure_onlyCA)         
        self.__representative_Rg(path_folder, path_tmp, ensembl_id, pdb_structure)
        self.__pdbselect(path_folder, ensembl_id, folder_pdb, pdb_structure)
        
        del pdb_structure
        gc.collect()
         
        self.__superimpose_multiple(path_folder, folder_data, ensembl_id, pdb_structure_onlyCA)
        
        self.__global(path_folder, folder_data, ensembl_id)        
        self.__chain(path_folder, folder_data, ensembl_id)
                
        self.__pdb2cif(path_maxit, pdb_content, ensembl_id, path_folder, folder_cif, path_tmp)
        
        
    def __radius_and_dmax(self, path_folder, folder_data, ensembl_id, pdb_structure_onlyCA, window_size=5):
        name_gg = os.path.join(folder_data, ensembl_id + "_gyration_global.txt")
        f_outgg = open(os.path.join(path_folder, name_gg), "w")
        
        name_gc = os.path.join(folder_data, ensembl_id + "_gyration_chain.txt")
        f_outgc = open(os.path.join(path_folder, name_gc), "w")
        
        name_gw = os.path.join(folder_data, ensembl_id + "_gyration_window.txt")
        f_outgw = open(os.path.join(path_folder, name_gw), "w")
        
        name_dg = os.path.join(folder_data, ensembl_id + "_dmax_global.txt")
        f_outdg = open(os.path.join(path_folder, name_dg), "w")
        
        name_dw = os.path.join(folder_data, ensembl_id + "_dmax_data.txt")
        f_outdw = open(os.path.join(path_folder, name_dw), "w")
       
        #header
        f_outgg.write("\t".join(["model", "gyration"]))
        f_outgc.write("\t".join(["model", "chain", "gyration"]))
        f_outgw.write("\t".join(["model", "chain", "window", "gyration"]))
        f_outdg.write("\t".join(["model", "dmax"]))
        f_outdw.write("\t".join(["model", "chain", "dmax"]))      
        
        for model in pdb_structure_onlyCA:            
            atom_coordinates = {
                c_chain.id: [x.coord for x in sorted(c_chain.get_atoms(), key=lambda x: x.id)]
                for c_chain in model.get_chains()
            }
            for chain_id, atoms in atom_coordinates.items():
                f_outgc.write("\n" + "\t".join([str(model.id + 1), chain_id, str(self._calculate_gyr_radius(atoms))]))
                for i, x in enumerate(self._create_window_slides(atoms, window_size), 1):
                    f_outgw.write("\n" + "\t".join([str(model.id + 1), chain_id, str(i), str(self._calculate_gyr_radius(x))])) 
            
            f_outgg.write("\n" + "\t".join([str(model.id + 1), str(self._calculate_gyr_radius([x.coord for x in sorted(model.get_atoms(), key=lambda x: x.id)]))]))
            
            f_outdg.write("\n" + "\t".join([str(model.id + 1), str(self._dmax_ca(model.get_atoms()))]))
            for c_chain in model.get_chains():
                f_outdw.write("\n" + "\t".join([str(model.id + 1), c_chain.id, str(self._dmax_ca(c_chain.get_atoms()))]))
        f_outgg.close()  
        f_outgc.close() 
        f_outgw.close() 
        f_outdg.close()    
        f_outdw.close() 
        self.__files.update({
            "radius_gyration_global" : name_gg,
            "radius_gyration_full" : name_gc,
            "radius_gyration_window": name_gw,
            "dmax_ca_global" : name_dg,
            "dmax_ca_chain" : name_dw
        })
        
    def _calculate_gyr_radius(self, atoms):
        atoms = [a for a in atoms]
        mass_centre = reduce(add, atoms) / len(atoms)
        gyr = reduce(
            add,
            [pow(distance.euclidean(x, mass_centre), 2) for x in atoms]
        )
        return sqrt(gyr / len(atoms))
    
    def _create_window_slides(self, elements, window_size=5):
        min_index = 0
        max_index = len(elements)-1
        slides = []
        for x in range(len(elements)):
            min_diff = min(min(abs(x-min_index), abs(x-max_index)), int((window_size-1)/2))
            x_s = x-min_diff
            x_e = x+min_diff+1
            slides.append(elements[x_s:x_e])
        return slides  
    
    def __superimpose_multiple(self, path_folder, folder_data, ensembl_id, pdb_structure_onlyCA):
        
        name_rg = os.path.join(folder_data, ensembl_id + "_rmsd_global.txt")
        f_outrg = open(os.path.join(path_folder, name_rg), "w")
        name_rw = os.path.join(folder_data, ensembl_id + "_rmsd_data.txt")
        f_outrw = open(os.path.join(path_folder, name_rw), "w")
                
        f_outrg.write("\t".join(["model1", "model2", "rmsd"]))
        f_outrw.write("\t".join(["model1", "model2", "chain", "rmsd"]))
        
        sup = Superimposer() 
        chains = [c_chain.id for c_chain in pdb_structure_onlyCA[0].get_chains()]
        for mod1_id in range(0, len(pdb_structure_onlyCA) - 1): 
            atoms_a = list(pdb_structure_onlyCA[mod1_id].get_atoms())
            atoms_chain_a = [list(pdb_structure_onlyCA[mod1_id][chain].get_atoms()) for chain in chains]
            for mod2_id in range(mod1_id + 1, len(pdb_structure_onlyCA)) :
                atoms_b = list(pdb_structure_onlyCA[mod2_id].get_atoms())
                atoms_chain_b = [list(pdb_structure_onlyCA[mod2_id][chain].get_atoms()) for chain in chains]
                rmsd_d = None
                if len(atoms_a) == len(atoms_b):
                    sup.set_atoms(atoms_a, atoms_b)
                    rmsd_d = float(sup.rms)
                f_outrg.write("\n" + "\t".join([str(mod1_id + 1), str(mod2_id + 1), str(rmsd_d)]))  
                for x in range(len(chains)): 
                    rms_temp = None
                    if len(atoms_chain_a[x]) == len(atoms_chain_b[x]):
                        sup.set_atoms(atoms_chain_a[x], atoms_chain_b[x])
                        rms_temp = float(sup.rms)
                    f_outrw.write("\n" + "\t".join([str(mod1_id + 1), str(mod2_id + 1), chains[x], str(rms_temp)])) 
        f_outrg.close()    
        f_outrw.close() 
        self.__files.update({
            "rmsd_global": name_rg,
            "rmsd_chain": name_rw,
        }) 
        
    def _dmax_ca(self, atom_list):
        dmax = 0
        for at1, at2 in combinations(atom_list, 2):
            dmax = max(dmax, at1 - at2)
        return float(dmax)
    
    def __global(self, path_folder, folder_data, ensembl_id):    
        name_f = os.path.join(folder_data, ensembl_id + "_other_summmary_global.txt")
        f_out = open(os.path.join(path_folder, name_f), "w")
        #header
        f_out.write("\t".join(["metric", "total", "mean", "stdev", "var", "min", "q1", "median", "q3", "max"]))           
        df = pd.read_csv(os.path.join(path_folder, self.__files["radius_gyration_global"]), sep="\t")
        d = stat_data(df["gyration"], "Radius of gyration")          
        f_out.write("\n" + "\t".join([str(sd) if sd != None else "" for sd in d ]))
        
        df = pd.read_csv(os.path.join(path_folder, self.__files["radius_gyration_window"]), sep="\t")
        d = stat_data(df["gyration"], "per residue Rg")          
        f_out.write("\n" + "\t".join([str(sd) if sd != None else "" for sd in d ]))
        
        df = pd.read_csv(os.path.join(path_folder, self.__files["dmax_ca_global"]), sep="\t")
        d = stat_data(df["dmax"], "Dmax CA - Global")          
        f_out.write("\n" + "\t".join([str(sd) if sd != None else "" for sd in d ]))
                
        df = pd.read_csv(os.path.join(path_folder, self.__files["dmax_ca_chain"]), sep="\t")
        d = stat_data(df["dmax"], "Dmax CA - Chain")          
        f_out.write("\n" + "\t".join([str(sd) if sd != None else "" for sd in d ]))
        
        df = pd.read_csv(os.path.join(path_folder, self.__files["rmsd_global"]), sep="\t")
        d = stat_data(df["rmsd"], "RMSD global")          
        f_out.write("\n" + "\t".join([str(sd) if sd != None else "" for sd in d ]))
        
        df = pd.read_csv(os.path.join(path_folder, self.__files["rmsd_chain"]), sep="\t")
        d = stat_data(df["rmsd"], "RMSD chain")          
        f_out.write("\n" + "\t".join([str(sd) if sd != None else "" for sd in d ]))
        
        f_out.close()
        self.__files.update({
            "other_global": name_f
        })   
    
    def __chain(self, path_folder, folder_data, ensembl_id):
        name_f = os.path.join(folder_data, ensembl_id + "_other_summmary_chain.txt")
        f_out = open(os.path.join(path_folder, name_f), "w")
        #header
        f_out.write("\t".join(["chain", "metric", "total", "mean", "stdev", "var", "min", "q1", "median", "q3", "max"]))
        
        df_max =  pd.read_csv(os.path.join(path_folder, self.__files["dmax_ca_chain"]), sep="\t")
        df_rmsd = pd.read_csv(os.path.join(path_folder, self.__files["rmsd_chain"]), sep="\t")
        df_rg = pd.read_csv(os.path.join(path_folder, self.__files["radius_gyration_full"]), sep="\t")
        chains = pd.unique(df_rg["chain"])
        
        for chain in chains:
            ds = df_max[df_max["chain"] == chain]            
            d = stat_data(ds["dmax"].tolist(), "Dmax")
            f_out.write("\n" + "\t".join([chain] + [str(sd) if sd != None else "" for sd in d]))
            
        for chain in chains:
            ds = df_rmsd[df_rmsd["chain"] == chain]            
            d = stat_data(ds["rmsd"].tolist(), "Rmsd")
            f_out.write("\n" + "\t".join([chain] + [str(sd) if sd != None else "" for sd in d]))
               
        for chain in chains:
            ds = df_rg[df_rg["chain"] == chain]            
            d = stat_data(ds["gyration"].tolist(), "Rg")
            f_out.write("\n" + "\t".join([chain] + [str(sd) if sd != None else "" for sd in d]))
               
        f_out.close()
        self.__files.update({
            "other_chain": name_f
        })
    
    
    def __representative_Rg(self, path_folder, path_tmp, ensembl_id, pdb_structure, ):
        name_f = os.path.join(path_tmp, ensembl_id + "_rep10_MC_Rg.pdb")  
        df = pd.read_csv(os.path.join(path_folder, self.__files["radius_gyration_global"]), sep="\t")
        #select 10 models by monte carlo
        df.sort_values('gyration', ascending=True, inplace=True)
        df = list(df["model"])  
        list_m = df        
                
        if len(pdb_structure) > 10:
            x = np.array_split(np.array(df), 10)
            list_m = [int(np.random.choice(k, 1)) for k in x]
            
        self.__files.update({
            "rep10_MC_Rg": name_f,
            "rep10_MC_Rg_mod": list_m
        })  
        pdbio = PDBIO()
        pdbio.set_structure(pdb_structure)
        class SelectModels(Select):
            def accept_atom(self, atom):
                return True
            def accept_chain(self, chain):
                return True
            def accept_residue(self, residue):
                return True
            def accept_model(self, model):
                return (model.id + 1) in list_m 
        pdbio.save(name_f, select=SelectModels())                   
                         
    def __pdb2cif(self, path_maxit, path_ens, ensembl_id, path_folder, folder_cif, path_tmp):
        #all models
        new_file = ensembl_id + ".cif"
        path_new = os.path.join(path_tmp, new_file)
        os.system(path_maxit + " -input " + path_ens + " -output " + path_new + " -o 1")
        
        cwd = os.getcwd() 
        os.chdir(path_tmp) 
        zip_file = new_file + ".tar.gz"
        if os.path.exists(zip_file) == True:
            os.remove(zip_file)
        tar = tarfile.open(zip_file, 'x:gz')
        tar.add(new_file)
        tar.close()
        #remove the cif all models
        os.remove(new_file)
        os.chdir(cwd)         
        #move the cif.tar.gz to folder_cif
        shutil.move(os.path.join(path_tmp, zip_file), os.path.join(path_folder, folder_cif, zip_file))
        
        #representative Rg
        name_f = os.path.join(folder_cif, ensembl_id + "_rep10_MC_Rg.cif")  
        os.system(path_maxit + " -input " + self.__files["rep10_MC_Rg"] + " -output " + os.path.join(path_folder, name_f) + " -o 1")
        #remove the pdb with the representative, was created before in other function and put in the temp folder
        os.remove(self.__files["rep10_MC_Rg"])
        
        self.__files.update({
            "all_models_cif": os.path.join(folder_cif, zip_file),
            "rep10_MC_Rg": name_f
        })
    
    def __pdbselect(self, path_folder, ensembl_id, folder_pdb, pdb_structure):    
        #always generate the pdb with the first model
        list_models = [1] + self.__files["rep10_MC_Rg_mod"]
        list_models = set(list_models)
        i = 0
        class SelectModel(Select):
            def accept_atom(self, atom):
                return True
            def accept_chain(self, chain):
                return True
            def accept_residue(self, residue):
                return True
            def accept_model(self, model):
                return (model.id + 1) == i 
        pdbio = PDBIO()
        pdbio.set_structure(pdb_structure)       
        for i in list_models:
            name_f = os.path.join(folder_pdb, ensembl_id + "_model_" + str(i) + ".pdb")        
            pdbio.save(os.path.join(path_folder, name_f), select=SelectModel()) 
            self.__files.update({
                "model_" + str(i): name_f
            })  
    
    def get_files_data(self):
        return self.__files    
            
class SelectCA(Select):
    """
    Select alpha carbon atoms.
    """
    def accept_atom(self, atom):
        return atom.name == "CA"
    def accept_chain(self, chain):
        return True
    def accept_residue(self, residue):
        return True
    def accept_model(self, model):
        return True  