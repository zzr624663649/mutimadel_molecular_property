import multiprocessing
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles, MACCSkeys, AllChem
from torch_geometric.data import Data, Dataset, DataLoader
from copy import deepcopy
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]
#分子指纹编码
def global_maccs_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    global_maccs_list = np.array(maccs).tolist()
    # 选择负/正样本比例小于1000且大于0.001的数据
    selected_index = [3, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165]
    selected_global_list = [global_maccs_list[x] for x in selected_index]
    return selected_global_list
def global_ecfp4_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    global_ecfp4_list = np.array(ecfp4).tolist()
    return global_ecfp4_list
def global_rdkit_des_data(smiles):
    descriptors_name = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt',
                           'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons',
                           'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
                           'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0',
                           'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n',
                           'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1',
                           'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
                           'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
                           'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',
                           'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',
                           'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
                           'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10',
                           'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                           'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
                           'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                           'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
                           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                           'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                           'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR',
                           'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH',
                           'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine',
                           'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                           'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                           'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide',
                           'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
                           'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
                           'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                           'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                           'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                           'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                           'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
                           'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
                           'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
                           'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
    m = Chem.MolFromSmiles(smiles)
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_name)
    descriptors = np.array(desc_calc.CalcDescriptors(m)).tolist()
    return descriptors


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    # return ' '.join(tokens)
    return tokens
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_labels(atom, use_chirality=True):
    results = one_of_k_encoding(atom.GetDegree(),
                                [0, 1, 2, 3, 4, 5, 6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()] \
              + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]
    atom_labels_list = np.array(results).tolist()
    atom_selected_index = [1, 2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20, 21]
    atom_labels_selected = [atom_labels_list[x] for x in atom_selected_index]
    return atom_labels_selected

def construct_input_from_smiles(smiles, max_len=200, global_feature='MACCS'):
    try:
        # built a pretrain data from smiles
        atom_list = []
        atom_token_list = ['c', 'C', 'O', 'N', 'n', '[C@H]', 'F', '[C@@H]', 'S', 'Cl', '[nH]', 's', 'o', '[C@]',
                           '[C@@]', '[O-]', '[N+]', 'Br', 'P', '[n+]', 'I', '[S+]',  '[N-]', '[Si]', 'B', '[Se]', '[other_atom]']
        all_token_list = ['[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F', '[C@@H]', '-', 'S', '/', 'Cl', '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]', '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]', '[Se]', '[other_atom]', '[other_token]']

        # 构建token转化成idx的字典
        word2idx = {}
        for i, w in enumerate(all_token_list):
            word2idx[w] = i
        # 构建token_list 并加上padding和global
        token_list = smi_tokenizer(smiles)
        padding_list = ['[PAD]' for x in range(max_len-len(token_list))]
        tokens = ['[GLO]'] + token_list + padding_list
        mol = MolFromSmiles(smiles)

        atom_example = mol.GetAtomWithIdx(0)
        atom_labels_example = atom_labels(atom_example)
        atom_mask_labels = [2 for x in range(len(atom_labels_example))]
        atom_labels_list = []
        atom_mask_list = []

        index = 0
        tokens_idx = []
        for i, token in enumerate(tokens):
            if token in atom_token_list:
                atom = mol.GetAtomWithIdx(index)
                an_atom_labels = atom_labels(atom)
                atom_labels_list.append(an_atom_labels)
                atom_mask_list.append(1)
                index = index + 1
                tokens_idx.append(word2idx[token])
            else:
                if token in all_token_list:
                    atom_labels_list.append(atom_mask_labels)
                    tokens_idx.append(word2idx[token])
                    atom_mask_list.append(0)
                elif '[' in list(token):
                    atom = mol.GetAtomWithIdx(index)
                    tokens[i] = '[other_atom]'
                    an_atom_labels = atom_labels(atom)
                    atom_labels_list.append(an_atom_labels)
                    atom_mask_list.append(1)
                    index = index + 1
                    tokens_idx.append(word2idx['[other_atom]'])
                else:
                    tokens[i] = '[other_token]'
                    atom_labels_list.append(atom_mask_labels)
                    tokens_idx.append(word2idx['[other_token]'])
                    atom_mask_list.append(0)
        if global_feature == 'MACCS':
            global_label_list = global_maccs_data(smiles)
        elif global_feature == 'ECFP4':
            global_label_list = global_ecfp4_data(smiles)
        elif global_feature == 'RDKIT_des':
            global_label_list = global_rdkit_des_data(smiles)

        tokens_idx = [word2idx[x] for x in tokens]
        if len(tokens_idx) == max_len + 1:
            return tokens_idx, global_label_list, atom_labels_list, atom_mask_list, smiles
        else:
            return 0, 0, 0, 0, 0
    except:
        return 0, 0, 0, 0, 0

def build_maccs_pretrain_data_and_save(smiles_list, output_smiles_path, global_feature='MACCS'):
    smiles_list = smiles_list
    tokens_idx_list = []
    global_label_list = []
    atom_labels_list = []
    atom_mask_list = []
    smile_list = []
    for i, smiles in enumerate(smiles_list):
        tokens_idx, global_labels, atom_labels, atom_mask, smiles = construct_input_from_smiles(smiles,
                                                                                        global_feature=global_feature)
        if tokens_idx != 0:
            tokens_idx_list.append(tokens_idx)
            global_label_list.append(global_labels)
            atom_labels_list.append(atom_labels)
            atom_mask_list.append(atom_mask)
            smile_list.append(smiles)
            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))
        else:
            print('{} is transformed failed!'.format(smiles))
    pretrain_data_list = [tokens_idx_list, global_label_list, atom_labels_list, atom_mask_list, smile_list]
    pretrain_data_np = np.array(pretrain_data_list)
    np.save(output_smiles_path, pretrain_data_np)


task_name = 'CHEMBL'
if __name__ == "__main__":
    n_thread = 8
    data = pd.read_csv('/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'.csv')
    smiles_list = data['smiles'].values.tolist()
    # 避免内存不足，将数据集分为10份来计算
    for i in range(10):
        n_split = int(len(smiles_list)/10)
        smiles_split = smiles_list[i*n_split:(i+1)*n_split]

        n_mol = int(len(smiles_split)/8)

        # creating processes
        p1 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[:n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+1)+'.npy'))
        p2 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[n_mol:2*n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+2)+'.npy'))
        p3 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[2*n_mol:3*n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+3)+'.npy'))
        p4 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[3*n_mol:4*n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+4)+'.npy'))
        p5 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[4*n_mol:5*n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+5)+'.npy'))
        p6 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[5*n_mol:6*n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+6)+'.npy'))
        p7 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[6*n_mol:7*n_mol],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+7)+'.npy'))
        p8 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[7*n_mol:],
                                                                                '/home/ubuntu/zzr/smiles_bertZ/data/pretrain_data/'+task_name+'_maccs_'+str(i*8+8)+'.npy'))

        # starting my_scaffold_split 1&2
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()

        # wait until my_scaffold_split 1&2 is finished
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()


        # both processes finished
        print("Done!")


