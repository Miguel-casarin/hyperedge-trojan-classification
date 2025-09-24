import logging
from najaeda import naja
import collections
import sys
import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from najaeda import netlist, naja
import csv

class Incidence_and_weights():
    Incidence_and_weights = {}

class Features_gates():
  features_gates = {}

class Vetorial_space():
  gates_vetorial = {}

class Features_hyperedges():
    features_hyperedges = {} 

class Hyperedge_weights():
    hyperedge_weights = {}   

designs_folder = "designs" 
gabaritos_folder = "gabaritos"

trojan_exemples = [
                    "design0.v",
                    "design1.v",
                    "design2.v",
                    "design3.v", 
                    "design4.v",
                    "design5.v",
                    "design6.v",
                    "design7.v",
                    "design8.v",
                    "design9.v",
                    "design10.v",
                    "design11.v",
                    "design12.v",
                    "design13.v",
                    "design14.v",
                    "design15.v",
                    "design16.v",
                    "design17.v",
                    "design18.v",
                    "design19.v"
                
                   ]

prefix_list = [ "0",  "1",  "2", "3", "4", "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"
              ]

NUMERIC_KEYS = [
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

def load_design(design_file, prim_file="iccadPrim.py"):
  netlist.reset()
  netlist.load_primitives_from_file(prim_file)
  top = netlist.load_verilog(design_file)
  universe = naja.NLUniverse.get()
  design = universe.getTopDesign()

  return design

# Pego o nome do nó para definir a chave da aresta
def get_key_edg(node):
    key = node.replace("g", "e")
    return key

def extract_numeric_features(all_features):
    numeric_features = {}
    for instance_name, data in all_features.items():
        # Inicializa o dicionário para a instância atual
        numeric_features[instance_name] = {}

        # Itera sobre as chaves e adiciona ao novo dicionário
        for key in NUMERIC_KEYS:  
            # Verifica se a chave existe e se o valor é numérico antes de adicionar
            if key in data:
                if isinstance(data[key], (int)):
                    numeric_features[instance_name][key] = data[key]
                # Lida com casos especiais como 'inf'
                elif data[key] >= 1e100:
                    numeric_features[instance_name][key] = -1 # Ou outro valor que represente 'infinito'
    return numeric_features

def vector_node(node, gates_vetorial):
    node_data = gates_vetorial.get(node, {})
    vector = [node_data.get(key, 0) for key in NUMERIC_KEYS]
    return vector

def features_hyperedg(gates_vetorial, size_k):
    Features_hyperedges.features_hyperedges = {}
    Hyperedge_weights.hyperedge_weights = {}
    
    features_hyperedgs = Features_hyperedges.features_hyperedges
    hyperedge_weights = Hyperedge_weights.hyperedge_weights


    nodes_list = list(gates_vetorial.keys()) 
    node_vectors = {node: vector_node(node, gates_vetorial) for node in nodes_list}

    # calcula a distancia
    node_vectors_array = np.array(list(node_vectors.values()))
    distance_matrix = euclidean_distances(node_vectors_array, node_vectors_array)
    #print(f"valor da distancia {distance_matrix}")

    # calcula sigma^2
    sigma_squared = np.mean(distance_matrix)
    if sigma_squared == 0:
        sigma_squared = 1e-6  

    for i, node in enumerate(nodes_list):
        key = get_key_edg(node)  # chave com prefxo

        # encontra os k vizinhos mais proximos
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

# Verificando o numero de acertos
def check_hits(detected_gates, gabarito_trojans):
    detected_gates = set(detected_gates)
    gabarito_trojans = set(gabarito_trojans)

    total_true = 0
    total_false = 0

    for gate in detected_gates:
        g_index = gate.find('g')
        if g_index != -1:
            gate_without_prefix = gate[g_index:]
        else:
            gate_without_prefix = gate
        
        if gate_without_prefix in gabarito_trojans:
            total_true += 1
        else:
            total_false += 1

    return total_true, total_false

def percentage (part, hole):
    try:
        return (part/hole)*100
    except:
        return 0

def calculate_totals(data_info):
    total_hits_true = 0
    total_errors = 0

    for design_id, result in data_info.items():
       total_hits_true += result['hits']
       total_errors += result['errors']

    return total_hits_true, total_errors

def process_all_designs():
    # Criar instâncias das classes
    features_g = Features_gates()
    features_v = Vetorial_space()
    
    # Limpar os dicionários das classes
    features_g.features_gates = {}
    features_v.gates_vetorial = {}

    # Processa cada arquivo separadamente
    for key, design_file in enumerate(trojan_exemples):
        design_path = f"{designs_folder}/{design_file}"
        
        try:
            # Carrega o design
            design = load_design(design_path, prim_file="iccadPrim.py")
            
            print(f"\nCarregando {design_file}")

            # Inicializa as features para o design atual
            features = initialize_features(design)
            annotate_features(design, features)
            annotate_features_cells(design, features)

            # Extrai as features numéricas para o design atual
            numeric_features = extract_numeric_features(features)

            # Salva as features no dicionário da classe, garantindo que cada arquivo tenha suas próprias chaves
            features_g.features_gates[key] = numeric_features
            
        except Exception as e:
            print(f"Erro ao processar {design_file}: {e}")
            continue

    # Mescla as informações de todos os arquivos no espaço vetorial
    for id_file, gates in features_g.features_gates.items():
        for gate_key, gate_f in gates.items():
            # Cria uma chave única para cada gate, combinando o id do arquivo e a chave do gate
            new_key = f"{id_file}{gate_key}"

            # Verifica se a chave já existe antes de adicionar
            if new_key not in features_v.gates_vetorial:
                features_v.gates_vetorial[new_key] = gate_f

    return features_g, features_v

def automated_trojan_detection(trojan_exemples, prefix_list, size_k, number_trojan_hyperedges, metodo, number_unique):
    results_dict = {}
    
    for design_id, design_file in enumerate(trojan_exemples):
        print(f"\n{'='*60}")
        print(f"PROCESSANDO DESIGN {design_id}: {design_file}")
        print(f"{'='*60}")
        
        # 1. Coleta gabarito do design atual COM FILTRO
        design_prefix = str(design_id)
        gabarito_path = f"{gabaritos_folder}/result{design_id}.txt"
        gates_unknown = []
        
        try:
            with open(gabarito_path, 'r') as f:
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
            print(f"Arquivo gabarito não encontrado: {gabarito_path}")
            gates_unknown = []

        # 2. Coleta trojans de exemplo COM FILTRO
        trojan_gates_list = []
        print(f"\nColetando trojans de exemplo...")
        
        for i in prefix_list:
            if i == design_prefix:  # Pula o design atual
                continue
            
            try:
                gabarito_file = f"{gabaritos_folder}/result{i}.txt"
                with open(gabarito_file, 'r') as f:
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
                            prefixed_gates = f"{i}{line}"
                            trojan_gates_list.append(prefixed_gates)
            except FileNotFoundError:
                print(f"Arquivo não encontrado: {gabarito_file}")
                continue

        print(f"Total de trojans de exemplo coletados: {len(trojan_gates_list)}")

        # Gera hiperarestas usando o espaço vetorial global
        print("Gerando hiperarestas...")
        features_hyperedges, hyperedge_weights = features_hyperedg(features_v.gates_vetorial, size_k)

        # Busca por trojans no design target
        design_file_prefix = f"{design_id}g"
        print(f"Buscando trojans para design prefix: {design_file_prefix}")

        # Escolhe o método de busca
        if metodo == 1:
            candidates = find_trojan(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, design_file_prefix)
        elif metodo == 2:
            candidates = mult_design_find_t(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, design_file_prefix, number_unique)
        else:
            print(f"Método {metodo} não reconhecido. Usando método 1 como padrão.")
            candidates = find_trojan(features_hyperedges, trojan_gates_list, number_trojan_hyperedges, design_file_prefix)

        # Lista gates encontrados
        incidence_list, total_incidence_unknown = list_gates(candidates, design_file_prefix)

        # Calcula os acertos
        total_true, total_false = check_hits(incidence_list, gates_unknown)

        # Salva resultados
        if metodo == 1:
            results_dict[design_id] = {
                'design_file': design_file,
                'gabarito_trojans': gates_unknown,
                'gabarito_count': len(gates_unknown),
                'hypereds candidates': len(candidates),
                'detected_gates': incidence_list,
                'suspicios_gates': total_incidence_unknown,
                'hits': total_true,
                'errors': total_false,
                'candidates_details': candidates
            }
        elif metodo == 2:
            unique_count = candidates[0]["unique_designs_count"] if candidates else 0
            results_dict[design_id] = {
                'design_file': design_file,
                'gabarito_trojans': gates_unknown,
                'gabarito_count': len(gates_unknown),
                'hypereds candidates': len(candidates),
                "unique_designs_count": unique_count,
                'detected_gates': incidence_list,
                'suspicios_gates': total_incidence_unknown,
                'hits': total_true,
                'errors': total_false,
                'candidates_details': candidates
            }

        # Limpa os dicionários das classes para a próxima iteração
        Features_hyperedges.features_hyperedges = {}
        Hyperedge_weights.hyperedge_weights = {}

    return results_dict



# Processa todos os designs e cria o espaço vetorial global
features_g, features_v = process_all_designs()

size_k = 10
number_trojan_hyperedges = 10
metodo = 1 # 1 normal - 2 leva em conta os unique designs
number_unique = 2
name_txt = "10trojan.txt"

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
        print(f"Hits corretos: {result['hits']}")
        print(f"Hits incorretos: {result['errors']}")
    print("="*60)

def make_txt(data_info):
    try:
        with open(name_txt, "w") as f:
            f.write(f"Trojans exemplos: {trojan_exemples}\n")
            f.write(f"Size k (número de vizinhos): {size_k}\n")
            f.write(f"Número mínimo de trojans por hiperaresta: {number_trojan_hyperedges}\n")
            f.write(f"Método de busca: {'Normal' if metodo == 1 else 'Com unique designs'}\n")
            if metodo == 2:
                f.write(f"Número mínimo de designs únicos por hiperaresta: {number_unique}\n")
            f.write("\n")
            for design_id, result in data_info.items():
                f.write(f"Design file: {design_id}\n")
                f.write(f"Gabarito trojans: {result['gabarito_trojans']}\n")
                f.write(f"Gabarito count: {result['gabarito_count']}\n")
                if metodo == 2:
                    f.write(f"Unique designs: {result['unique_designs_count']}\n")
                f.write(f"Candidates found: {result['hypereds candidates']}\n")
                f.write(f"Detected gates: {result['detected_gates']}\n")
                f.write(f"Suspicios gates: {result['suspicios_gates']}\n")
                f.write(f"Hits corretos: {result['hits']}\n")
                f.write(f"Hits incorretos: {result['errors']}\n\n")
        print("TXT relatorio gerado")
    except Exception as e:
        print(f"ERROR ON MAKE TXT: {e}")


def export_results_csv(data_info, number_trojan_hyperedges, csv_filename):
    header = [
        "Design",
        "N° de trojans gabarito",
        "N° trojans por hiperaresta",
        "N° de gates suspeitos",
        "N° acertos",
        "N° erros",
    ]
    with open(csv_filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for design_id in sorted(data_info.keys()):
            r = data_info[design_id]
            w.writerow([
                design_id,
                r.get("gabarito_count", 0),
                number_trojan_hyperedges,
                r.get("suspicios_gates", 0),
                r.get("hits", 0),
                r.get("errors", 0),
            ])
    return csv_filename

make_txt(data_info)
csv_name = f"{number_trojan_hyperedges}trojan.csv"
saved = export_results_csv(data_info, number_trojan_hyperedges, csv_name)
print(f"CSV gerado: {saved}")

"""
for i in range(1, 10):
    size_k = 10
    number_trojan_hyperedges = i
    number_unique = 2
    metodo = 1 # 1 normal - 2 leva em conta os unique designs

    data_info = automated_trojan_detection(trojan_exemples, prefix_list, size_k, number_trojan_hyperedges, metodo, number_unique)

    # Calcula o hit_rate e escreve no arquivo
    hit_rate_results = hit_rate(data_info)
    
    if metodo == 1:
        write_hit_rate_results_to_file(hit_rate_results, i, metodo, number_trojan_hyperedges)
    
    if metodo == 2:
        write_hit_rate_results_to_file(hit_rate_results, i, metodo, number_trojan_hyperedges, number_unique)

print("\nTodos os resultados foram salvos em hit_rate_results.txt")
"""