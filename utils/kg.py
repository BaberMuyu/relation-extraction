import json
import copy
import re
from global_config import *


class KG(object):
    def __init__(self):
        self.graph = {}
        self.entities = {}
        self.list_0 = ('主题曲', '父亲', '母亲', '注册资本', '气候', '朝代', '面积', '总部地点',
                       '修业年限', '所在城市', '改编自', '成立日期', '海拔', '国籍', '祖籍', '专业代码', '邮政编码',
                       '首都')
        self.list_0_meanless = ('票房', '人口数量', '占地面积')  # 涉及到数量train data准确率不高
        self.special_list = {'妻子'}  # '丈夫'
        self.conflict = 0

    def generate_kg(self, file):
        train_data = json.load(open(file, 'r', encoding='utf-8'))
        for dd in train_data:
            for sspo in dd['spo_list']:
                self.add_node(sspo)

    def add_node(self, spo):
        if spo['subject_type'] in self.entities.keys():
            self.entities[spo['subject_type']].append(spo['subject'])
        else:
            self.entities[spo['subject_type']] = [spo['subject']]
        for key in spo['object'].keys():
            if spo['object_type'][key] in self.entities.keys():
                self.entities[spo['object_type'][key]].append(spo['object'][key])
            else:
                self.entities[spo['object_type'][key]] = [spo['object'][key]]

        if spo['predicate'] in self.list_0:
            if spo['subject'] in self.graph.keys():
                if spo['predicate'] in self.graph[spo['subject']].keys():
                    if spo['object']['@value'] != self.graph[spo['subject']][spo['predicate']]:
                        self.conflict += 1
                        print('conflict!!new:{}, exist:{}'.format(spo['object']['@value'],
                                                                  self.graph[spo['subject']][spo['predicate']]))
                else:
                    self.graph[spo['subject']][spo['predicate']] = spo['object']['@value']
            else:
                self.graph[spo['subject']] = {spo['predicate']: spo['object']['@value']}
        else:
            if spo['subject'] in self.graph.keys():
                if spo['predicate'] in self.graph[spo['subject']].keys():
                    if spo['object']['@value'] not in self.graph[spo['subject']][spo['predicate']]:
                        self.graph[spo['subject']][spo['predicate']].append(spo['object']['@value'])
                else:
                    self.graph[spo['subject']][spo['predicate']] = [spo['object']['@value']]
            else:
                self.graph[spo['subject']] = {spo['predicate']: [spo['object']['@value']]}

    def check_spo(self, text, spo):
        def is_emperor(string):
            a = ['宗', '帝', '曹操', '祖', '皇']
            for aa in a:
                if string.find(aa) >= 0:
                    return True
            else:
                return False

        error_flag = False
        predicate = spo['predicate']
        new_spos = [spo]
        if predicate in self.list_0 and spo['subject'] in self.graph.keys() and predicate in self.graph[
            spo['subject']].keys():
            spo_value = spo['object']['@value']
            kg_value = self.graph[spo['subject']][spo['predicate']]
            if spo_value != kg_value:
                if text.find(kg_value) >= 0:
                    if spo['predicate'] in ('成立日期', '上映时间'):
                        spo['object']['@value'] = spo_value if len(spo_value) >= len(kg_value) else kg_value
                    else:
                        # print(spo, self.graph[spo['subject']][spo['predicate']])
                        spo['object']['@value'] = self.graph[spo['subject']][spo['predicate']]
                    # error_flag = True
        elif predicate in self.special_list and spo['subject'] in self.graph.keys() and \
                predicate in self.graph[spo['subject']].keys():
            spo_value = spo['object']['@value']
            kg_value = self.graph[spo['subject']][spo['predicate']]
            if is_emperor(spo['subject']):
                for kg_obj in kg_value:
                    if spo_value.find(kg_obj) >= 0:
                        spo['object']['@value'] = kg_obj
                    elif text.find(kg_obj) >= 0:
                        new_spo = copy.deepcopy(spo)
                        new_spo['object']['@value'] = kg_obj
                        # print('add', new_spo)
                        new_spos.append(new_spo)
                        error_flag = True
            elif len(kg_value) == 1 and text.find(kg_value[0]) >= 0:
                spo['object']['@value'] = kg_value[0]
        return error_flag, new_spos

    def load_kg(self, file):
        fp = open(file, 'r', encoding='utf-8')
        self.graph, self.entities = json.load(fp)

    def dump_kg(self, outfile):
        fp = open(outfile, 'w+', encoding='utf-8')
        json.dump([self.graph, self.entities], fp, ensure_ascii=False)


class Node(object):
    def __init__(self, predicate, node_name, is_sbj):
        self.name = node_name
        self.in_edges = {}
        self.out_edges = {}
        self.gender = 'unknown'
        self.professions = []
        self.fictional = False
        self.add_properity(predicate, is_sbj)

    def __str__(self):
        node_info = {'name': self.name,
                     'gender': self.gender,
                     'professions': self.professions,
                     'fictional': self.fictional,
                     'in': dict((rlt, [node.name for node in nodes]) for rlt, nodes in self.in_edges.items()),
                     'out': dict((rlt, [node.name for node in nodes]) for rlt, nodes in self.out_edges.items()),
                     }
        return str(node_info)

    def add_properity(self, predicate, is_sbj=True):
        def set_gender_(gender):
            if self.gender == 'unknown' or self.gender == gender:
                self.gender = gender
            else:
                self.gender = 'error'

        if is_sbj:
            if predicate == '妻子':
                set_gender_('male')
            if predicate in ('配音', '饰演'):
                self.professions.append(predicate)
        else:
            if predicate == '父亲':
                set_gender_('male')
            if predicate in ('妻子', '母亲'):
                set_gender_('female')
            if predicate in ('作者', '编剧', '歌手', '制片人', '作词', '导演', '作曲', '主演', '主持人'):
                self.professions.append(predicate)
            if predicate in ('主角', '配音', '角色', '饰演'):
                self.fictional = True

    def add_out_edge(self, edge_type, obj_node):
        if edge_type not in self.out_edges.keys():
            self.out_edges[edge_type] = [obj_node]
        elif obj_node not in self.out_edges[edge_type]:
            self.out_edges[edge_type].append(obj_node)
        else:
            return
        obj_node.add_in_edge(edge_type, self)

    def add_in_edge(self, edge_type, sbj_node):
        if edge_type not in self.in_edges.keys():
            self.in_edges[edge_type] = [sbj_node]
        elif sbj_node not in self.in_edges[edge_type]:
            self.in_edges[edge_type].append(sbj_node)

    def del_out_edge(self, edge_type, obj_node):
        if edge_type not in self.out_edges.keys():
            print('kg error')
        elif obj_node not in self.out_edges[edge_type]:
            print('kg error')
        else:
            del (self.out_edges[edge_type][self.out_edges[edge_type].index(obj_node)])
            if len(self.out_edges[edge_type]) == 0:
                del (self.out_edges[edge_type])
            obj_node.del_in_edge(edge_type, self)

    def del_in_edge(self, edge_type, sbj_node):
        if edge_type not in self.in_edges.keys():
            print('error')
        elif sbj_node not in self.in_edges[edge_type]:
            print('error')
        else:
            del (self.in_edges[edge_type][self.in_edges[edge_type].index(sbj_node)])
            if len(self.in_edges[edge_type]) == 0:
                del (self.in_edges[edge_type])

    def check_out_edge(self, edge_type, obj_node):
        if edge_type in self.out_edges.keys():
            if obj_node in self.out_edges[edge_type]:
                return 'CORRECT'
        return 'UNKNOWN'


class KnowledgeGraph(object):
    def __init__(self):
        self.graph = {}
        self.uniques = ('父亲', '母亲', '改编自', '国籍', '祖籍', '专业代码', '邮政编码', '妻子')
        self.conflicts = (('妻子', '母亲', '父亲'), ('嘉宾', '主持人'), ('主角', '主演'), ('角色', '主演'), ('作者', '主角'))
        self.conflicts_1 = ('歌手', '作词', '作曲')
    def add_spo(self, spo):
        def get_node(predicate, entity, is_sbj=True):
            if entity not in self.graph.keys():
                new_node = Node(predicate, entity, is_sbj)
                self.graph[entity] = new_node
            else:
                self.graph[entity].add_properity(predicate, is_sbj)
            return self.graph[entity]

        sbj_node = get_node(spo['predicate'], spo['subject'], is_sbj=True)
        obj_node = get_node(spo['predicate'], spo['object']['@value'], is_sbj=False)
        sbj_node.add_out_edge(spo['predicate'], obj_node)

    def load_graph_from_raw_data(self, files):
        for filename in files:
            fp = open(filename, 'r', encoding='utf-8')
            data = json.load(fp)
            for d in data:
                for spo in d['spo_list']:
                    self.add_spo(spo)
            fp.close()

    def load_graph_from_spos(self, spos):
        for spo in spos:
            self.add_spo(spo)

    def check_spo(self, text, spos):
        def correct_rlt(spo, sbj_node, obj_node):
            for conflict_pair in self.conflicts:
                if spo['predicate'] in conflict_pair:
                    for c_rlt in conflict_pair:
                        if c_rlt != spo['predicate']:
                            if c_rlt in sbj_node.out_edges.keys() and obj_node in sbj_node.out_edges[c_rlt]:
                                # print(sbj, rlt, obj, '-->', sbj, c_rlt, obj)
                                if c_rlt != '嘉宾':
                                    spo['predicate'] = c_rlt
                                    spo = correct_type(spo)
            if spo['predicate'] in self.conflicts_1:
                if spo['predicate'] not in obj_node.professions:
                    for p in self.conflicts_1:
                        if obj_node.professions.count(p) > 30:
                            spo['predicate'] = p
                            break
            return spo

        def correct_entity(spo):
            new_temp_spos = []
            sbj, rlt, obj = spo['subject'], spo['predicate'], spo['object']['@value']
            if spo['predicate'] in self.uniques:
                if rlt in sbj_node.out_edges.keys() and len(sbj_node.out_edges[rlt]) >= 1:
                    for oobj in sbj_node.out_edges[rlt]:
                        if len(oobj.name) > 1 and text.find(oobj.name) != -1:
                            if rlt != '妻子':
                                spo['object']['@value'] = oobj.name if oobj.name not in obj else obj
                                # print(sbj, rlt, obj, '-->', spo['object']['@value'])
                                break
                            else:
                                new_spo = copy.deepcopy(spo)
                                new_spo['object']['@value'] = oobj.name if oobj.name not in obj else obj
                                if new_spo not in new_temp_spos:
                                    new_temp_spos.append(new_spo)
                                if spo not in new_temp_spos and obj_node.in_edges.get('妻子') is None and len(
                                        sbj_node.out_edges[rlt]) > 1:
                                    new_temp_spos.append(spo)
            if not new_temp_spos:
                new_temp_spos.append(spo)
            return new_temp_spos

        new_spos = []
        for spo in spos:
            sbj, rlt, obj = spo['subject'], spo['predicate'], spo['object']['@value']
            sbj_node = self.graph.get(sbj)
            obj_node = self.graph.get(obj)
            if sbj_node and obj_node and sbj_node.check_out_edge(rlt, obj_node) == 'UNKNOWN':
                spo = correct_rlt(spo, sbj_node, obj_node)
                new_spos.extend(correct_entity(spo))
            else:
                new_spos.append(spo)
        spos = []
        for spo in new_spos:
            if spo not in spos:
                spos.append(spo)
        return spos

    def find_all_possible_spos(self, text, entities):
        spos = []
        for e in entities:
            if e in self.graph.keys():
                for edge_type in self.graph[e].in_edges.keys():
                    for node in self.graph[e].in_edges[edge_type]:
                        if node.name in entities:
                            spos.append([node.name, edge_type, e])
                for edge_type in self.graph[e].out_edges.keys():
                    for node in self.graph[e].out_edges[edge_type]:
                        if node.name in entities:
                            spos.append([e, edge_type, node.name])
        return spos

    def fix_spo(self, text, spo, force_flag=None):
        sbj = spo['subject']
        rlt = spo['predicate']
        obj = spo['object']['@value']
        sbj_node = self.graph.get(sbj)
        obj_node = self.graph.get(obj)
        spos = []
        if sbj_node is not None and obj_node is None and (
                force_flag == 'OBJ' or spo['object_type']['@value'] in ('人物', '历史人物', '娱乐人物')):
            kg_obj_nodes = sbj_node.out_edges.get(rlt)
            if kg_obj_nodes:
                for oobj in kg_obj_nodes:
                    if oobj.name in obj and len(oobj.name) > 1 and (
                            force_flag == 'OBJ' or (len(obj) > 4 and obj.find('·') == -1 and have_chinese(obj))):
                        new_spo = copy.deepcopy(spo)
                        new_spo['object']['@value'] = oobj.name
                        if new_spo not in spos:
                            # print(sbj, rlt, obj, ' --> ', oobj.name)
                            spos.append(new_spo)

        if sbj_node is None and obj_node is not None and (
                force_flag == 'SBJ' or spo['subject_type'] in ('人物', '历史人物', '娱乐人物')):
            kg_sbj_nodes = obj_node.in_edges.get(rlt)
            if kg_sbj_nodes:
                for ssbj in kg_sbj_nodes:
                    if ssbj.name in sbj and len(ssbj.name) > 1 and (
                            force_flag == 'SBJ' or (len(sbj) > 4 and sbj.find('·') == -1 and have_chinese(sbj))):
                        new_spo = copy.deepcopy(spo)
                        new_spo['subject'] = ssbj.name
                        if new_spo not in spos:
                            # print(sbj, rlt, obj, ' --> ', ssbj.name)
                            spos.append(new_spo)
        if len(spos) == 0:
            spos = [spo]
        return spos

    def self_check(self):
        global std_kg

        def correct_rlt_(sbj_node, obj_node):
            rlt_list = []
            for rlt in sbj_node.out_edges.keys():
                if obj_node in sbj_node.out_edges[rlt]:
                    rlt_list.append(rlt)
            for conflict_pair in self.conflicts:
                conflict_list = []
                for rlt in conflict_pair:
                    if rlt in rlt_list:
                        conflict_list.append(rlt)
                if len(conflict_list) == 2:
                    # print(conflict_pair)
                    # print(sbj_node)
                    # print(obj_node)
                    conflict_list = set(conflict_list)
                    if conflict_list == set(['母亲', '父亲']):
                        if obj_node.name in std_kg.graph.keys() and std_kg.graph[obj_node.name].gender == 'male':
                            sbj_node.del_out_edge('母亲', obj_node)
                        elif obj_node.name in std_kg.graph.keys() and std_kg.graph[obj_node.name].gender == 'female':
                            sbj_node.del_out_edge('父亲', obj_node)
                        elif sbj_node.name[0] == obj_node.name[0]:
                            sbj_node.del_out_edge('母亲', obj_node)
                        else:
                            sbj_node.del_out_edge('父亲', obj_node)
                            sbj_node.del_out_edge('母亲', obj_node)
                    elif conflict_list == set(['主角', '主演']):
                        if obj_node.name in std_kg.graph.keys() and '主演' in std_kg.graph[obj_node.name].professions:
                            sbj_node.del_out_edge('主角', obj_node)
                        elif obj_node.name in std_kg.graph.keys() and std_kg.graph[obj_node.name].fictional:
                            sbj_node.del_out_edge('主演', obj_node)
                        else:
                            sbj_node.del_out_edge('主角', obj_node)
                            sbj_node.del_out_edge('主演', obj_node)
                    elif conflict_list == set(['角色', '主演']):
                        if obj_node.name in std_kg.graph.keys() and '主演' in std_kg.graph[obj_node.name].professions:
                            sbj_node.del_out_edge('角色', obj_node)
                        elif obj_node.name in std_kg.graph.keys() and std_kg.graph[obj_node.name].fictional:
                            sbj_node.del_out_edge('主演', obj_node)
                        else:
                            sbj_node.del_out_edge('角色', obj_node)
                            sbj_node.del_out_edge('主演', obj_node)
                    elif conflict_list == set(['作者', '主角']):
                        if obj_node.name in std_kg.graph.keys() and '作者' in std_kg.graph[obj_node.name].professions:
                            sbj_node.del_out_edge('主角', obj_node)
                        elif obj_node.name in std_kg.graph.keys() and std_kg.graph[obj_node.name].fictional:
                            sbj_node.del_out_edge('作者', obj_node)
                        else:
                            sbj_node.del_out_edge('作者', obj_node)
                            sbj_node.del_out_edge('主角', obj_node)
                    elif conflict_list == set(['嘉宾', '主持人']):
                        if obj_node.name in std_kg.graph.keys() and '主持人' in std_kg.graph[obj_node.name].professions:
                            sbj_node.del_out_edge('嘉宾', obj_node)
                        else:
                            sbj_node.del_out_edge('嘉宾', obj_node)
                            sbj_node.del_out_edge('主持人', obj_node)
                    # print(sbj_node)
                    # print(obj_node)
                    # print('\n')
                elif len(conflict_list) > 2:
                    print(111)

        for sbj in self.graph.keys():
            sbj_node = self.graph[sbj]
            obj_nodes = []
            for rlt in sbj_node.out_edges.keys():
                for obj_node in sbj_node.out_edges[rlt]:
                    if obj_node not in obj_nodes:
                        obj_nodes.append(obj_node)
            for obj_node in obj_nodes:
                correct_rlt_(sbj_node, obj_node)

    def generate_spos(self):
        spos = []
        for sbj_node in self.graph.values():
            for rlt in sbj_node.out_edges.keys():
                for obj_node in sbj_node.out_edges[rlt]:
                    spos.append({
                        'subject_type': schemas_dict[rlt]['subject_type'],
                        'subject': sbj_node.name,
                        'predicate': rlt,
                        'object_type': {'@value': schemas_dict[rlt]['object_type']['@value']},
                        'object': {'@value': obj_node.name}
                    })
        return spos

    def check_spo_1(self, spo):
        sbj, rlt, obj = spo['subject'], spo['predicate'], spo['object']['@value']
        sbj_node = self.graph.get(sbj)
        obj_node = self.graph.get(obj)
        if sbj_node and obj_node and sbj_node.check_out_edge(rlt, obj_node) == 'CORRECT':
            return True
        else:
            return False


# class NewKnowledgeGraph(object):
#     def __init__(self):
#         self.graph = {}
#         self.uniques = ('父亲', '母亲', '改编自', '国籍', '祖籍', '专业代码', '邮政编码', '妻子')
#         self.conflicts = (('妻子', '母亲', '父亲'), ('嘉宾', '主持人'), ('主角', '主演'), ('角色', '主演'), ('作者', '主角'))
# 
#     def add_spo(self, spo):
#         def get_node(predicate, entity, is_sbj=True):
#             if entity not in self.graph.keys():
#                 new_node = Node(predicate, entity, is_sbj)
#                 self.graph[entity] = new_node
#             else:
#                 self.graph[entity].add_properity(predicate, is_sbj)
#             return self.graph[entity]
# 
#         sbj_node = get_node(spo['predicate'], spo['subject'], is_sbj=True)
#         obj_node = get_node(spo['predicate'], spo['object']['@value'], is_sbj=False)
#         sbj_node.add_out_edge(spo['predicate'], obj_node)
#         
#     def copy_node(self, node, new_name):
#         new_node = copy.deepcopy(node)
#         new_node.name = new_name
#         new_node.in_edges = {}
#         new_node.out_edges = {}
#         return new_node
# 
#     def combine_node(self, node_0, node_1):
#         for rlt in node_1.out_edges.keys():
#             for obj_node in node_1.out_edges[rlt]:
#                 node_0.add_out_edge(rlt, obj_node)
#                 node_1.del_out_edge(rlt, obj_node)
#         for rlt in node_0.in_edges.keys():
#             for sbj_node in node_0.in_edges[rlt]:
#                 sbj_node.add_out_edge(rlt, node_0)
#                 sbj_node.del_out_edge(rlt, node_1)
#         return node_0
# 
#     def insert_node(self, node_0, node_1):
#         for rlt in node_0.out_edges.keys():
#             for obj_node in node_0.out_edges[rlt]:
#                 obj_node.add_in_edge(rlt, node_1)
#         for rlt in node_0.in_edges.keys():
#             for sbj_node in node_0.in_edges[rlt]:
#                 sbj_node.add_out_edge(rlt, node_1)
# 
#     def self_check(self, std_graph):
#         def check_(sbj, obj):
#             std_sbj_node, std_obj_node = std_graph.get_node(node.name), std_graph.get_node(obj_node.name)
#             if std_sbj_node and std_obj_node and std_sbj_node.check_out_edge(rlt, std_obj_node) == 'correct':
#                 continue
#             else:
#                 if std_sbj_node and std_obj_node:
#                     pass
#                 elif std_sbj_node and not std_obj_node:
#                     pass
#                 elif std_obj_node and not std_sbj_node:
#                     pass
#                 else:
#                     pass
#         for node in self.graph.values():
#             for rlt in node.out_edges.keys():
#                 for obj_node in node.out_edges[rlt]:
#                     check_(node.name, obj_node.name)
#             for rlt in node.in_edges.keys():
#                 for sbj_node in node.in_edges[rlt]:
#                     check_(sbj_node.name, node.name)
# 
#     def split_nodes(self):
#         for node in self.graph.values()[:]:
#             nodes_name = []
#             for node_type in node.types:
#                 if node_type in ('文学作品', '作品', '影视作品', '图书作品', '歌曲'):
#                     nodes_name = re.split("》《|》、《|》，《|\d\d |》和《", node.name)
#                     break
#                 elif node.name.find('，') == -1:
#                     if node_type in ('企业', '企业/品牌', "机构"):
#                         nodes_name = re.split("、|/|､", node.name)
#                         break
#                     elif node_type in ('人物', '历史人物', '娱乐人物') and len(node.name) <= 40:
#                         nodes_name = re.split("、|/", node.name)
#                         if len(nodes_name) > 1:
#                             new_nodes_name = []
#                             for n in nodes_name[:]:
#                                 if len(n) < 15:
#                                     new_nodes_name.append(n)
#                             nodes_name = new_nodes_name
#                         break
# 
#             if nodes_name:
#                 for new_node_name in nodes_name:
#                     new_node = self.copy_node(node, new_node_name)
#                     self.insert_node(node, new_node)
#                     if new_node_name in self.graph.keys():
#                         self.combine_node(self.graph[new_node_name], new_node)
#                     else:
#                         self.graph[new_node_name] = new_node
# 
#     def correct_unknown_node(self, std_graph):
#         def is_illegal_name(node_name):
#             return True
# 
#         def correct_(sbj, rlt, obj):
#             sbj_nodes_name = []
#             obj_nodes_name = []
#             std_sbj_node, std_obj_node = std_graph.get_node(sbj), std_graph.get_node(obj)
#             if std_sbj_node and not std_obj_node:
#                 sbj2obj_nodes = std_sbj_node.out_edges.get(rlt)
#                 if sbj2obj_nodes:
#                     for oobj in sbj2obj_nodes:
#                         if oobj.name in obj and len(oobj.name) > 1 and (is_illegal_name(obj) or False):
#                             obj_nodes_name.append(oobj.name)
#             elif not std_sbj_node and std_obj_node:
#                 obj2sbj_nodes = std_obj_node.in_edges.get(rlt)
#                 if obj2sbj_nodes:
#                     for ssbj in obj2sbj_nodes:
#                         if ssbj.name in sbj and len(sbj.name) > 1 and (is_illegal_name(sbj) or False):
#                             sbj_nodes_name.append(ssbj.name)
#             return sbj_nodes_name, obj_nodes_name
# 
# 
# 
#         for sbj_node in self.graph.values():
#             for rlt in sbj_node.out_edges.keys():
#                 for obj_node in sbj_node.out_edges[rlt]:
#                     sbj, obj = sbj_node.name, obj_node.name
#                     correct_(sbj, obj)
# 
# 
#     def generate_spos(self):
#         spos = []
#         for sbj_node in self.graph.values():
#             for rlt in sbj_node.out_edges.keys():
#                 for obj_node in sbj_node.out_edges[rlt]:
#                     spos.append({
#                         'subject_type': schemas_dict['rlt']['subject_type'],
#                         'subject': sbj_node.name,
#                         'predicate': rlt,
#                         'object_type': {'@value': schemas_dict['rlt']['object_type']['@value']},
#                         'object': {'@value': obj_node.name}
#                     })
#         return spos

std_kg = KnowledgeGraph()
std_kg.load_graph_from_raw_data([ROOT_DATA + 'lic_2020/decomposed_dev_data.json',
                                 ROOT_DATA + 'lic_2020/decomposed_train_data.json'])

file_name = ROOT_DATA + 'lic_2020/my_schema.json'
fp = open(file_name, mode='r', encoding='utf-8')
schemas = [json.loads(d) for d in fp.readlines()]
schemas_dict = dict([(d['predicate'], d) for d in schemas])


def correct_type(spo):
    spo['subject_type'] = schemas_dict[spo['predicate']]['subject_type']
    spo['object_type']['@value'] = schemas_dict[spo['predicate']]['object_type']['@value']
    return spo


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def have_chinese(string):
    for s in string:
        if is_chinese(s):
            return True
    return False


if __name__ == '__main__':
    from global_config import *

    train_file = ROOT_DATA + 'lic_2020/decomposed_train_data.json'
    dev_file = ROOT_DATA + 'lic_2020/decomposed_dev_data.json'
    test_file = ROOT_RESULT + 'ave3_test_joint.json'
    test_file_out = ROOT_RESULT + 'kg_test_joint.json'

    kg = KnowledgeGraph()
    kg.load_graph_from_raw_data([train_file, dev_file])
    # kg.self_check()
    exit()
    # kg.generate_kg(train_file)
    # kg.dump_kg(kg_file)
    # exit()
    # kg.load_kg(kg_file)

    count = 0
    new_data = []
    exist_num = 0
    unknown_num = 0
    unknown_num_0 = 0
    unknown_num_1 = 0
    all_num = 0
    with open(test_file, 'r', encoding='utf-8') as fp:
        raw_data = fp.readlines()
        for d in raw_data:
            d = json.loads(d)
            spos = kg.check_spo(d['text'], d['spo_list'])
            if spos != d['spo_list']:
                # print(spos)
                pass
    print(exist_num, unknown_num, unknown_num_0, all_num)
