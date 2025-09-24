# Dependencias 
import logging
from najaeda import naja
import collections
import sys
import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from najaeda import netlist, naja

class Incidence_and_weights():
    Incidence_and_weights = {}


designs_folder = "designs" 
gabaritos_folder = "gabaritos"

trojan_exemples = [
                    "design0.v",
                    "design1.v",
                    "design2.v",
                    "design3.v"


                   ]
prefix_list = [ "0",  "1",  "2", "3"]

def constructDFF(lib):
  cell = naja.SNLDesign.createPrimitive(lib, "dff")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Output, "Q")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "RN")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "SN")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "CK")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "D")

def load(db):
  logging.info("Loading verilog built-in + iccad primitives")
  lib = naja.NLLibrary.createPrimitives(db, "iccad")
  constructDFF(lib)


def initialize_features(design, trojans = [], inputDesign = False):
    # Init features for design
    features = {}
    all_insts = list(design.getInstances())

    for inst in all_insts:
        inst_name = inst.getName()

        fanin = 0
        fanout = 0
        for instTerm in inst.getInstTerms():
            direction = instTerm.getDirection()
            if (direction == naja.SNLTerm.Direction.Output):
                net = instTerm.getNet()
                for sinkTerm in net.getInstTerms():
                    sink_direction = sinkTerm.getDirection()
                    if (sink_direction == naja.SNLTerm.Direction.Input):
                        fanout += 1
            else:
                fanin += 1

        isTrojan = 0
        if inst_name in trojans:
            isTrojan = 1
        if inputDesign:
            isTrojan = -1

        features[inst_name] = {
            'fanin': fanin,
            'fanout': fanout,
            'closestInputDepth': float(sys.float_info.max),
            'closestInputName': [],
            'closestDriveFFDepth': float(sys.float_info.max),
            'closestDriveFFName': [],
            'closestOutputDepth': float(sys.float_info.max),
            'closestOutputName': [],
            'closestSinkFFDepth': float(sys.float_info.max),
            'closestSinkFFName': [],
            'ratioFan2': float(sys.float_info.max),
            'ratioFan3': float(sys.float_info.max),
            'ratioFan4': float(sys.float_info.max),
            'ratioFan5': float(sys.float_info.max),
            'inFF2': float(sys.float_info.max),
            'inFF3': float(sys.float_info.max),
            'inFF4': float(sys.float_info.max),
            'inFF5': float(sys.float_info.max),
            'outFF2': float(sys.float_info.max),
            'outFF3': float(sys.float_info.max),
            'outFF4': float(sys.float_info.max),
            'outFF5': float(sys.float_info.max),
            'inInv2': float(sys.float_info.max),
            'inInv3': float(sys.float_info.max),
            'inInv4': float(sys.float_info.max),
            'inInv5': float(sys.float_info.max),
            'outInv2': float(sys.float_info.max),
            'outInv3': float(sys.float_info.max),
            'outInv4': float(sys.float_info.max),
            'outInv5': float(sys.float_info.max),
            'label': isTrojan
        }

    return features

def annotate_features_cells(design, features):
    for inst in design.getInstances():
      annotate_cell(design, features, inst)

def annotate_cell(design, features, inst):
    inst_name = inst.getName()
    cellsIn = [[] for i in range(5)]
    cellsOut = [[] for i in range(5)]

    # Save cell names from up to 5 levels behind instance
    q_back = collections.deque()
    q_back.append((inst, 0))
    visited_back = {inst}
    while q_back:
        current_inst, depth = q_back.popleft()
        if depth >= 5: continue
        cellsIn[depth].append(current_inst.getModel().getName())
        for in_term in current_inst.getInstTerms():
            if in_term.getDirection() == naja.SNLTerm.Direction.Input:
                in_net = in_term.getNet()
                if not in_net: continue
                for term_on_net in in_net.getInstTerms():
                    if term_on_net.getDirection() == naja.SNLTerm.Direction.Output:
                        prev_inst = term_on_net.getInstance()
                        if prev_inst not in visited_back:
                            q_back.append((prev_inst, depth + 1))
                            visited_back.add(prev_inst)

    # Save cell names from up to 5 levels after instance
    q_forward = collections.deque()
    q_forward.append((inst, 0))
    visited_forward = {inst}
    while q_forward:
        current_inst, depth = q_forward.popleft()
        if depth >= 5: continue
        cellsOut[depth].append(current_inst.getModel().getName())
        for out_term in current_inst.getInstTerms():
            if out_term.getDirection() == naja.SNLTerm.Direction.Output:
                out_net = out_term.getNet()
                if not out_net: continue
                for term_on_net in out_net.getInstTerms():
                    if term_on_net.getDirection() == naja.SNLTerm.Direction.Input:
                        next_inst = term_on_net.getInstance()
                        if next_inst not in visited_forward:
                            q_forward.append((next_inst, depth + 1))
                            visited_forward.add(next_inst)

    # computing ratioFan
    for i in range(1, 5): # levels 2 through 5
        level_x = i + 1
        fanin_levelX_count = len(cellsIn[i])
        ratio = 0
        if features[inst_name]['fanout'] != 0:
            ratio = fanin_levelX_count / features[inst_name]['fanout']
        features[inst_name][f'ratioFan{level_x}'] = ratio

    # computing counting inv/not and dffs
    cumulative_in_ffs = 0
    cumulative_in_invs = 0
    for i in range(5):
        level_cells = cellsIn[i]
        cumulative_in_ffs += sum(1 for cell_name in level_cells if "dff" in cell_name)
        cumulative_in_invs += sum(1 for cell_name in level_cells if "not" in cell_name)

        # paper doesn't use 0 (level 1)
        if i >= 1:
            level_x = i + 1
            features[inst_name][f'inFF{level_x}'] = cumulative_in_ffs
            features[inst_name][f'inInv{level_x}'] = cumulative_in_invs

    cumulative_out_ffs = 0
    cumulative_out_invs = 0
    for i in range(5):
        level_cells = cellsOut[i]
        cumulative_out_ffs += sum(1 for cell_name in level_cells if "dff" in cell_name)
        cumulative_out_invs += sum(1 for cell_name in level_cells if "not" in cell_name)

        if i >= 1:
            level_x = i + 1
            features[inst_name][f'outFF{level_x}'] = cumulative_out_ffs
            features[inst_name][f'outInv{level_x}'] = cumulative_out_invs

def _update_path_feature(inst_features, depth_key, name_key, new_depth, new_names):
    # Check if path needs an update
    is_better = new_depth < inst_features[depth_key]
    is_same = new_depth == inst_features[depth_key]

    if is_better:
        inst_features[depth_key] = new_depth
        inst_features[name_key] = list(new_names) # copy
    elif is_same:
        for name in new_names:
            if name not in inst_features[name_key]:
                inst_features[name_key].append(name)

    return is_better or is_same

def _process_instance(inst, features, comb_depth, comb_names, ff_depth, ff_name, is_forward):
    inst_name = inst.getName()
    inst_features = features[inst_name]

    # Determine keys based on traversal direction
    comb_depth_key = 'closestInputDepth' if is_forward else 'closestOutputDepth'
    comb_name_key = 'closestInputName' if is_forward else 'closestOutputName'
    ff_depth_key = 'closestDriveFFDepth' if is_forward else 'closestSinkFFDepth'
    ff_name_key = 'closestDriveFFName' if is_forward else 'closestSinkFFName'

    # Update features and check if we should continue traversal from this instance
    comb_path_updated = _update_path_feature(inst_features, comb_depth_key, comb_name_key, comb_depth, comb_names)
    ff_path_updated = _update_path_feature(inst_features, ff_depth_key, ff_name_key, ff_depth, [ff_name] if ff_name else [])

    if not comb_path_updated and not ff_path_updated:
        return None # Prune search if no path was improved

    # Determine next state for fanout/fanin nets
    is_ff = "dff" in inst.getModel().getName()
    next_ff_name = inst_name if is_ff else ff_name
    next_ff_depth = 0 if is_ff else ff_depth

    return (inst_features[comb_name_key], next_ff_name, next_ff_depth)


def annotate_features(design, features):
    # Forwards and Backwards BFS to annotate features
    q = collections.deque()
    q_back = collections.deque()

    # Keep track of the best depths seen for each net to avoid redundant processing
    net_min_depths = {} # (net, is_forward) -> min_depth

    # Initialize queues with primary inputs and outputs
    for term in design.getBitTerms():
        is_input = term.getDirection() == naja.SNLTerm.Direction.Input
        net = term.getNet()
        if not net: continue

        # State is net, depth to port, port names, ff names, ff depth
        state = (net, 0, [str(net)], None, float(sys.float_info.max))
        if is_input:
            q.append(state)
            net_min_depths[(net, True)] = 0
        else:
            q_back.append(state)
            net_min_depths[(net, False)] = 0

    # Forward propagation
    while q:
        current_net, pi_depth, pi_names, last_ff, depth_from_ff = q.popleft()

        for term in current_net.getInstTerms():
            inst = term.getInstance()

            # To get the next cell, check the terms that are Inputs (so continue from any other)
            if (term.getDirection() == naja.SNLTerm.Direction.Output):
                continue

            process_result = _process_instance(inst, features, pi_depth + 1, pi_names, (depth_from_ff + 1) if last_ff else float('inf'), last_ff, is_forward=True)

            if process_result:
                new_pi_names, next_ff_name, next_depth_from_ff = process_result

                # Enqueue fanout nets
                for out_term in inst.getInstTerms():
                    if (out_term.getDirection() == naja.SNLTerm.Direction.Output):
                        out_net = out_term.getNet()
                        if not out_net: continue

                        new_pi_depth = pi_depth + 1
                        if new_pi_depth < net_min_depths.get((out_net, True), float('inf')):
                            net_min_depths[(out_net, True)] = new_pi_depth
                            q.append((out_net, new_pi_depth, new_pi_names, next_ff_name, next_depth_from_ff))

    # Backward propagationprint("Starting backwards propagation...")
    while q_back:
        current_net, po_depth, po_names, last_ff, depth_from_ff = q_back.popleft()

        for term in current_net.getInstTerms():
            inst = term.getInstance()

            # To get the next cell, check the terms that are Outputs (so continue from any other)
            if (term.getDirection() == naja.SNLTerm.Direction.Input):
                continue

            process_result = _process_instance(inst, features, po_depth + 1, po_names, (depth_from_ff + 1) if last_ff else float('inf'), last_ff, is_forward=False)

            if process_result:
                new_po_names, next_ff_name, next_depth_from_ff = process_result

                # Enqueue fanin nets
                for in_term in inst.getInstTerms():
                    if (in_term.getDirection() == naja.SNLTerm.Direction.Input):
                        in_net = in_term.getNet()
                        if not in_net: continue

                        new_po_depth = po_depth + 1
                        if new_po_depth < net_min_depths.get((in_net, False), float('inf')):
                            net_min_depths[(in_net, False)] = new_po_depth
                            q_back.append((in_net, new_po_depth, new_po_names, next_ff_name, next_depth_from_ff))

    print("--> Annotation complete.")

def extract_numeric_features(all_features):
    numeric_features = {}
    for instance_name, data in all_features.items():
        # Inicializa o dicionário para a instância atual
        numeric_features[instance_name] = {}

        # Lista de chaves numéricas a serem extraídas
        numeric_keys = [
            'fanin',
            'fanout',
            'closestInputDepth',
            'closestDriveFFDepth',
            'closestOutputDepth',
            'closestSinkFFDepth',
            'ratioFan2',
            'ratioFan3',
            'ratioFan4',
            'ratioFan5',
            'inFF2',
            'inFF3',
            'inFF4',
            'inFF5',
            'outFF2',
            'outFF3',
            'outFF4',
            'outFF5',
            'inInv2',
            'inInv3',
            'inInv4',
            'inInv5',
            'outInv2',
            'outInv3',
            'outInv4',
            'outInv5',
        ]



        # Itera sobre as chaves e adiciona ao novo dicionário
        for key in numeric_keys:
            # Verifica se a chave existe e se o valor é numérico antes de adicionar
            if key in data:
                # O `label` já é numérico. Para as outras, garanta que sejam
                if isinstance(data[key], (int)):
                    numeric_features[instance_name][key] = data[key]
                # Lida com casos especiais como 'inf'
                elif data[key] >= 1e100:
                    numeric_features[instance_name][key] = -1 # Ou outro valor que represente 'infinito'
    return numeric_features

class Features_gates():
  features_gates = {}

class Vetorial_space():
  gates_vetorial = {}

def load_design(design_file, prim_file="iccadPrim.py"):
  netlist.reset()
  netlist.load_primitives_from_file(prim_file)
  top = netlist.load_verilog(design_file)
  universe = naja.NLUniverse.get()
  design = universe.getTopDesign()

  return design


features_hyperedgs = {}
hyperedge_weights  = {}


# Pego o nome do nó para definir a chave da aresta
def get_key_edg(node):
    key = node.replace("g", "e")
    return key


def vector_node(node, gates_vetorial):
    feature_keys = [
        'fanin',
        'fanout',
        'closestInputDepth',
        'closestDriveFFDepth',
        'closestOutputDepth',
        'closestSinkFFDepth',
        'ratioFan2',
        'ratioFan3',
        'ratioFan4',
        'ratioFan5',
        'inFF2',
        'inFF3',
        'inFF4',
        'inFF5',
        'outFF2',
        'outFF3',
        'outFF4',
        'outFF5',
        'inInv2',
        'inInv3',
        'inInv4',
        'inInv5',
        'outInv2',
        'outInv3',
        'outInv4',
        'outInv5',
    ]

    # Access the specific node's features from gates_vetorial
    node_data = gates_vetorial.get(node, {})

    # Create the feature vector
    vector = [node_data.get(key, 0) for key in feature_keys]
    return vector

def features_hyperedg(gates_vetorial, size_k):
    global features_hyperedgs
    global hyperedge_weights

    nodes_list = list(gates_vetorial.keys())  # Use the keys from gates_vetorial as nodes
    node_vectors = {node: vector_node(node, gates_vetorial) for node in nodes_list}

    # Calculate the distance matrix efficiently
    node_vectors_array = np.array(list(node_vectors.values()))
    distance_matrix = euclidean_distances(node_vectors_array, node_vectors_array)

    # Calculate sigma^2
    sigma_squared = np.mean(distance_matrix)
    if sigma_squared == 0:
        sigma_squared = 1e-6  # Ensure sigma^2 is not zero

    for i, node in enumerate(nodes_list):
        key = get_key_edg(node)  # Generate the edge key

        # Find the k-nearest neighbors
        k_nearest_indices = np.argsort(distance_matrix[i])[1:size_k + 1]
        hyperedge = [node] + [nodes_list[j] for j in k_nearest_indices]
        features_hyperedgs[key] = hyperedge

        total_weight = 0
        for node_j_index in k_nearest_indices:
            distance = distance_matrix[i, node_j_index]
            weight_ij = np.exp(- (distance**2) / (2 * sigma_squared))
            total_weight += weight_ij

        hyperedge_weights[key] = total_weight

    return features_hyperedgs, hyperedge_weights

def find_trojan(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, unknown_design):
    candidates = []
    trojan_set = set(trojan_gates_list)

    for edge_key, nodes in features_hyperedges.items():
        found_trojans = list(set(nodes) & trojan_set)
        trojan_count = len(found_trojans)

        has_unknown = any(node.startswith(unknown_design) for node in nodes)
        design_gates = [node for node in nodes if node.startswith(unknown_design)]

        if trojan_count >= number_trojan_hyperedges and has_unknown:
            number_unknown = sum(1 for node in nodes if node.startswith(unknown_design))
            info = {
                "edge_key": edge_key,
                "trojan_count": trojan_count,
                "count unknown": number_unknown,
                "found_trojans": found_trojans,
                "design_gates": design_gates,

                "hyperedge": nodes
            }
            candidates.append(info)


    return candidates



# Buscando trojans em hipearestas com designs diferentes
# vou precisar arrumar a lógica do not_prefix, (precisa ser atualizada dentro do for )
def mult_design_find_t(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, unknown_design, number_unique): 
    candidates = []
    trojan_set = set(trojan_gates_list)

    for edge_key, nodes in features_hyperedges.items():
        found_trojans = list(set(nodes) & trojan_set)
        trojan_count = len(found_trojans)

        # lógica que procura por gates de designs diferentes 
        unique_designs = set(node.split("g")[0] for node in found_trojans if node.split("g")[0] != unknown_design)
        unique_count = len(unique_designs)

        has_unknown = any(node.startswith(unknown_design) for node in nodes)
        design_gates = [node for node in nodes if node.startswith(unknown_design)]

        # precisa ter um número de trojans maior ou igual ao defido e a mesma lógica para designs diferentes dentro das hipearestas 
        if trojan_count >= number_trojan_hyperedges and has_unknown and unique_count >= number_unique: 
            number_unknown = sum(1 for node in nodes if node.startswith(unknown_design))
            info = {
                "edge_key": edge_key,
                "trojan_count": trojan_count,
                "unique_designs_count": unique_count,
                "count unknown": number_unknown,
                "found_trojans": found_trojans,
                "design_gates": design_gates,
                "hyperedge": nodes
            }
            candidates.append(info)

    return candidates

 
# conta sem repetições os gates do design desconhecido
def list_gates(candidates, unknown_design):
    gate_list = []
    count = 0

    for hyperedge in candidates:
        for node in hyperedge["hyperedge"]:
            if node.startswith(unknown_design) and node not in gate_list:
                gate_list.append(node)
                count += 1

    return gate_list, count

def classifica(candidates, unknown_design):
    # Inicializa o dicionário global da classe
    classification = Incidence_and_weights.Incidence_and_weights

    for hyperedge in candidates:
        edge_key = hyperedge["edge_key"]
        trojan_count = hyperedge["trojan_count"]
        for node in hyperedge["hyperedge"]:
            if node.startswith(unknown_design):
                if node not in classification:
                    classification[node] = {
                        "count": 0,
                        "trojan_sum": 0,
                        "edges": []
                    }
                classification[node]["count"] += 1
                classification[node]["trojan_sum"] += trojan_count
                classification[node]["edges"].append(edge_key)

    return classification

""" 

Automatizando a chamada das funções
Agora eu entendo o pq de seguir boas práticas 

"""

def automated_trojan_detection(trojan_exemples, prefix_list, size_k, number_trojan_hyperedges, metodo, number_unique):
  
    results_dict = {}
    
    # Processa cada design individualmente
    # O "design_id" serve como chave para salvar as informações
    for design_id, desig_file in enumerate(trojan_exemples):
        design_prefix = str(design_id)
        print(f"\n{'='*60}")
        print(f"PROCESSANDO DESIGN {design_id}: {desig_file}")
        print(f"{'='*60}")
        
        # 1. Coleta trojans conhecidos (de todos os outros arquivos)
        trojan_gates_list = []
        
        for i in prefix_list:
            if i == design_prefix:
                continue
            result_file = f"{gabaritos_folder}/result{i}.txt"
            
            try:
                with open(result_file, "r") as f:
                    inside_block = False
                    for line in f:
                        line = line.strip()
                        if line == "TROJAN_GATES":
                            inside_block = True
                            continue
                        elif line == "END_TROJAN_GATES":
                            inside_block = False
                            continue
                        if inside_block and line.startswith("g"):
                            prefixed = f"{i}{line}"
                            trojan_gates_list.append(prefixed)
            except FileNotFoundError:
                print(f"Arquivo não encontrado: {result_file}")
                continue
        
        # 2. Coleta gabarito do arquivo target (para comparação posterior)
        unknown_result = f"{gabaritos_folder}/result{design_prefix}.txt"
        gates_unknown = []
        
        try:
            with open(unknown_result, "r") as f:
                inside_block = False
                for line in f:
                    line = line.strip()
                    if line == "TROJAN_GATES":
                        inside_block = True
                        continue
                    elif line == "END_TROJAN_GATES":
                        inside_block = False
                        continue
                    if inside_block and line.startswith("g"):
                        gates_unknown.append(line)
        except FileNotFoundError:
            print(f"Arquivo gabarito não encontrado: {unknown_result}")
            gates_unknown = []
        

        # 3. Processa features de todos os arquivos
        features_g = Features_gates()
        features_v = Vetorial_space()
        
        print("\nProcessando features de todos os designs...")
        for key, design_file in enumerate(trojan_exemples):
            design_path = f"{designs_folder}/{design_file}"
            try:
                design = load_design(design_path, prim_file="iccadPrim.py")
                print(f"Carregando design {key}: {design_file}")
                
                # Inicializa as features para o design atual
                features = initialize_features(design)
                annotate_features(design, features)
                annotate_features_cells(design, features)
                
                # Extrai as features numéricas para o design atual
                numeric_features = extract_numeric_features(features)
                
                # Salva as features no dicionário
                features_g.features_gates[key] = numeric_features
                
            except Exception as e:
                print(f"Erro ao processar {design_file}: {e}")
                continue
        
        # 4. Mescla as informações de todos os arquivos no espaço vetorial
        for id_file, gates in features_g.features_gates.items():
            for gate_key, gate_f in gates.items():
                new_key = f"{id_file}{gate_key}"
                if new_key not in features_v.gates_vetorial:
                    features_v.gates_vetorial[new_key] = gate_f
        
        
        # 5. Gera hiperarestas

        features_hyperedges, hyperedge_weights = features_hyperedg(features_v.gates_vetorial, size_k)
    
        # 6. Busca por trojans no design target
        design_file_prefix = f"{design_id}g"
        print(f"\nBuscando trojans para design prefix: {design_file_prefix}")
        
        # 7. Escolhe qual o critério para buscar os tojans
        if metodo == 1:
            candidates = find_trojan(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, design_file_prefix)
            classifica(candidates, design_file_prefix)
            

        if metodo == 2:
            candidates = mult_design_find_t(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, design_file_prefix, number_unique)
        
        
        # 8. Lista gates encontrados
        incidence_list, total_incidence_unknown = list_gates(candidates, design_file_prefix)
        
        
        # 9. Salva resultados
        if metodo == 1:
            results_dict[design_id] = {
                'design_file': design_file,
                'gabarito_trojans': gates_unknown,
                'gabarito_count': len(gates_unknown),
                'hypereds candidates': len(candidates),
                'detected_gates': incidence_list,
                'suspicios_gates': total_incidence_unknown,
                'candidates_details': candidates  # Gurda as informações das hiperaretas candidatas
            }
        
        if metodo == 2:
            

            # Pega o unique_count do primeiro candidato, se houver, senão define como 0
            if candidates and "unique_designs_count" in candidates[0]:
                unique_count = candidates[0]["unique_designs_count"]
            else:
                unique_count = 0

            results_dict[design_id] = {
                'design_file': design_file,
                'gabarito_trojans': gates_unknown,
                'gabarito_count': len(gates_unknown),
                'hypereds candidates': len(candidates),
                "unique_designs_count": unique_count,
                'detected_gates': incidence_list,
                'suspicios_gates': total_incidence_unknown,
                'candidates_details': candidates  # Gurda as informações das hiperaretas candidatas
            }
        
        """
        print(f"\nResumo para design {design_id}:")
        
        print(f"- Total de trojans do design gabarito: {len(gates_unknown)}")
        print(f"- Incidencia do design desconhecido: {total_incidence_unknown}")

        """
        
    return results_dict

size_k = 10
number_trojan_hyperedges = 5
metodo = 1  # 1 normal - 2 leva em conta os unique designs
number_unique = 3

data_info = automated_trojan_detection(trojan_exemples, prefix_list, size_k, number_trojan_hyperedges, metodo, number_unique)

def print_data(data_info):
    for design_id, result in data_info.items():
        print(f"\n{'='*60}")
        print(f"RESULTADOS PARA DESIGN {design_id}:")
        print(f"{'='*60}")
        print(f"Gabarito trojans: {result['gabarito_trojans']}")
        print(f"Gabarito count: {result['gabarito_count']}")
        print(f"Hypereds: {result['hypereds candidates']}")
        print(f"Detected gates: {result['detected_gates']}")
        print(f"Suspicios gates: {result['suspicios_gates']}")
    
def make_txt (data_info):
    try:
        with open("relatorio2.txt", "w") as f:
            for design_id, result in data_info.items():
                f.write(f"Design file: {design_id}\n")
                f.write(f"Gabarito trojans: {result['gabarito_trojans']}\n")
                f.write(f"Gabarito count: {result['gabarito_count']}\n")
                f.write(f"Unique designs:{result['unique_designs_count']}\n")
                f.write(f"Candidates found: {result['hypereds candidates']}\n")
                f.write(f"Detected gates: {result['detected_gates']}\n")
                f.write(f"Suspicios gates: {result['suspicios_gates']}\n\n")
        print("TXT relatorio gerado")
    except:
        print("ERROR ON MAKE TXT")

for gate, info in classifica.items():
                print(f"Gate: {gate}")
                print(f"  Quantas vezes apareceu: {info['count']}")
                print(f"  Soma dos trojans das hiperarestas: {info['trojan_sum']}")
                print(f"  Hiperarestas: {info['edges']}")
                print("-" * 40)

#make_txt(data_info)

#print(data_info)

# Supondo que você tem 'candidates' e 'design_file_prefix' definidos
result_classifica = classifica(candidates, design_file_prefix)
for gate, info in result_classifica.items():
    print(f"Gate: {gate}")
    print(f"  Quantas vezes apareceu: {info['count']}")
    print(f"  Soma dos trojans das hiperarestas: {info['trojan_sum']}")
    print(f"  Hiperarestas: {info['edges']}")
    print("-" * 40)