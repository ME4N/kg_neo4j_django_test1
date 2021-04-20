from django.shortcuts import render
from django.http import HttpResponse
import json
import time
from django.contrib.auth.decorators import login_required
import re
import json
import json
import sys
import os
from py2neo import authenticate, Graph, Node
import jieba
from kg.ner_project import main_lstm
from kg.rela_project import predict_rela
from py2neo import Node, Relationship, size, order, Graph, NodeSelector
import numpy as np
import neo4j

authenticate("localhost:7474", "neo4j", "123456")
graph = Graph("http://localhost:7474")

def add_entity(request):
	ctx = {}
	success={}
	if (request.GET):
		entity1 = request.GET['entity1_text']
		relation = request.GET['relation_name_text']
		entity2 = request.GET['entity2_text']
		# print('entity1=', entity1)
		# print('relation=', relation)
		# print('entity2=', entity2)
		relation = relation.lower()
		end = graph.find_one(label="car_industry", property_key="name", property_value=entity2)
		start = graph.find_one(label="car_industry", property_key="name", property_value=entity1)
		answer=findEntityRelation(entity1,relation,entity2)
		if answer:
			ctx={'title' : '<h1>实体关系已存在</h1>'}
			return render(request, 'add_entity.html', {'ctx': ctx})
		if end == None:
			end = Node('car_industry', name=entity2)
		if start == None:
			start = Node('car_industry', name=entity1)
		r = Relationship(start,relation, end, name=relation)
		print(r)
		addResult=r
		graph.create(r)
		success = {'title' : '<h1>实体关系已添加</h1>'}
		return render(request,'add_entity.html',{'addResult':json.dumps(addResult,ensure_ascii=False),'ctx': ctx,'success':success})
	return render(request, 'add_entity.html', {'ctx': ctx})

def del_relation(request):
	ctx = {}
	success = {}
	if (request.GET):
		entity1 = request.GET['entity1_text']
		relation = request.GET['relation_name_text']
		entity2 = request.GET['entity2_text']
		relation = relation.lower()
		end = graph.find_one(label="car_industry", property_key="name", property_value=entity2)
		start = graph.find_one(label="car_industry", property_key="name", property_value=entity1)
		answer = findEntityRelation(entity1, relation, entity2)
		if len(answer)>0:
			graph.run("MATCH (n1 {name:\"" + str(entity1) + "\"})- [rel {name:\"" + str(
				relation) + "\"}] -> (n2{name:\"" + entity2 + "\"}) DELETE rel")
			success={'title' : '<h1>实体关系已删除</h1>'}
			return render(request, 'del_relation.html',
						  {'success': success})
		ctx={'title' : '<h1>实体关系不存在</h1>'}
	return render(request, 'del_relation.html',{'success': success,'ctx':ctx})

def neo4j_search(word, is_init=0):
	if not is_init:
		answer = graph.run("MATCH (n1:car_industry {name:\"" + word + "\"})- [rel] -> (n2) RETURN n1,rel,n2").data()
		print("answer=",answer)
		edges = []
		for res in answer:
			edges.append({"source": word,
						  "target": res['n2']['name'],
						  "relation": res['rel']['name'],
						  "label": 'relation',
						  "color_flag": -1})
		return edges


def getEntityRelationbyEntity(value):
	answer = graph.run(
		"MATCH (entity1) - [rel] -> (entity2)  WHERE entity1.name = \"" + str(value) + "\" RETURN rel,entity2").data()
	print("answer=", answer)
	return answer


def findRelationByEntity( entity1):
	answer = graph.run("MATCH (n1 {name:\"" + entity1 + "\"})- [rel] -> (n2) RETURN n1,rel,n2").data()
	print("answer=", answer)
	if(answer is None):
		answer = graph.run("MATCH (n1:NewNode {title:\""+entity1+"\"})- [rel] -> (n2) RETURN n1,rel,n2" ).data()
	return answer


# 查找entity2及其对应的关系
def findRelationByEntity2(entity1):
	answer = graph.run("MATCH (n1)- [rel] -> (n2 {name:\"" + str(entity1) + "\"}) RETURN n1,rel,n2").data()

	# if(answer is None):
	# 	answer = self.graph.run("MATCH (n1)- [rel] -> (n2:NewNode {title:\""+entity1+"\"}) RETURN n1,rel,n2" ).data()
	return answer


# 根据entity1和关系查找enitty2
def findOtherEntities(entity, relation):
	answer = graph.run("MATCH (n1 {name:\"" + str(entity) + "\"})- [rel {name:\"" + str(
		relation) + "\"}] -> (n2) RETURN n1,rel,n2").data()
	# if(answer is None):
	#	answer = self.graph.run("MATCH (n1:NewNode {title:\"" + entity + "\"})- [rel:RELATION {type:\""+relation+"\"}] -> (n2) RETURN n1,rel,n2" ).data()

	return answer


# 根据entity2和关系查找enitty1
def findOtherEntities2( entity, relation):
	answer = graph.run("MATCH (n1)- [rel {name:\"" + str(relation) + "\"}] -> (n2 {name:\"" + str(
		entity) + "\"}) RETURN n1,rel,n2").data()
	# if(answer is None):
	#	answer = self.graph.run("MATCH (n1)- [rel:RELATION {type:\""+relation+"\"}] -> (n2:NewNode {title:\"" + entity + "\"}) RETURN n1,rel,n2" ).data()

	return answer


# 根据两个实体查询它们之间的最短路径
def findRelationByEntities( entity1, entity2):
	'''answer = graph.run("MATCH (p1{name:\"" + str(entity1) + "\"}),(p2{name:\"" + str(
		entity2) + "\"}),p=shortestpath((p1)-[rel*]-(p2)) RETURN rel").evaluate()

	print('answer=',answer)'''
	answer=graph.run("MATCH (n1{name:\"" + str(entity1) + "\"}), (n2{name:\"" + str(entity2) + "\"}), p = shortestpath((n1)-[rel*]-(n2)) RETURN p").evaluate()
	print('type(answer)=',type(answer))
	print('answer=', answer)
	# answer = self.graph.run("MATCH (p1:HudongItem {title:\"" + entity1 + "\"})-[rel:RELATION]-(p2:HudongItem{title:\""+entity2+"\"}) RETURN p1,p2").data()


	# answer = self.graph.data("MATCH (n1:HudongItem {title:\"" + entity1 + "\"})- [rel] -> (n2:HudongItem{title:\""+entity2+"\"}) RETURN n1,rel,n2" )
	# if(answer is None):
	#	answer = self.graph.data("MATCH (n1:HudongItem {title:\"" + entity1 + "\"})- [rel] -> (n2:NewNode{title:\""+entity2+"\"}) RETURN n1,rel,n2" )
	# if(answer is None):
	#	answer = self.graph.data("MATCH (n1:NewNode {title:\"" + entity1 + "\"})- [rel] -> (n2:HudongItem{title:\""+entity2+"\"}) RETURN n1,rel,n2" )
	# if(answer is None):
	#	answer = self.graph.data("MATCH (n1:NewNode {title:\"" + entity1 + "\"})- [rel] -> (n2:NewNode{title:\""+entity2+"\"}) RETURN n1,rel,n2" )
	relationDict = []
	if (answer is not None):
		for x in answer:
			tmp = {}
			start_node = x.start_node()
			end_node = x.end_node()
			tmp['n1'] = start_node
			print('start_node=', start_node)
			print('end_node=',end_node)
			print('type(start_node)=',type(start_node))
			tmp['n2'] = end_node
			tmp['rel'] = x
			relationDict.append(tmp)


	print('relationDict=',relationDict)
	print('type(relationDict)=',type(relationDict))
	return relationDict
#	return answer

# 查询数据库中是否有对应的实体-关系匹配
def findEntityRelation( entity1, relation, entity2):
	answer = graph.run("MATCH (n1 {name:\"" + str(entity1) + "\"})- [rel {name:\"" + str(
		relation) + "\"}] -> (n2{name:\"" + entity2 + "\"}) RETURN n1,rel,n2").data()

	return answer

'''
def qindex(request):
	ctx = {}
	#根据传入的实体名称搜索出关系
	if(request.GET):
		q_word = request.GET['user_text']
		#连接数据库
		kg_search = getEntityRelationbyEntity(q_word)
		if len(kg_search) == 0:
			#若数据库中无法找到该实体，则返回数据库中无该实体
			ctx= {'title' : '<h1>数据库中暂未添加该实体</h1>'}
			return render(request,'kg_index_test.html',{'ctx':json.dumps(ctx,ensure_ascii=False)})
		else:
			#返回查询结果
			#将查询结果按照"关系出现次数"的统计结果进行排序
			#entityRelation = sortDict(entityRelation)

			return render(request,'kg_index_test.html',{'kg_search':json.dumps(kg_search,ensure_ascii=False)})

	return render(request,"kg_index_test.html",{'ctx':ctx})

'''
def qindex(request):
	"""用户搜索"""
	q_word = request.GET.get('q_word', '')
	kg_search={}
	kg_result = {}
	kg_result["edges"] = []
	back = request.GET.get('back', '')
	if q_word != "":
		kg_search = neo4j_search(q_word)
		kg_result["edges"].extend(kg_search)
		ff = open('./data/search_data.csv', 'w', encoding='utf-8')
		ff.write(json.dumps(kg_result))

	if back == '1':
		open('./data/search_data.csv', 'w', encoding='utf-8')
		ff = open('./kg/data/ld.txt', encoding='utf-8')
		num = 0
		for line in ff.readlines():
			word = line.strip('\n').split("$$")[0]
			kg_result["edges"].append({"source": word, "target": word})
			num += 1
			if num > 100:
				break
		print('kg_result=',kg_result)
	print('kg_result=', kg_result)
	print('kg_search=', kg_search)
	return render(request, 'kg_index.html', {'kg_result': json.dumps(kg_result)})




def click_word(request):
	click_word = request.GET.get("click_word", "")
	jsondata = {}
	if click_word != "":
		try:
			jsondata = json.loads(open('./data/search_data.csv', 'r', encoding='utf-8').read())
		except Exception as e:
			print(e)
		kg_search = neo4j_search(click_word)
		if len(jsondata) != 0:
			jsondata['edges'].extend(kg_search)
		else:
			jsondata['edges'] = kg_search
		ff = open('./data/search_data.csv', 'w', encoding='utf-8')
		ff.write(json.dumps(jsondata))

	return render(request, 'kg_index.html', {'kg_result': json.dumps(jsondata)})


def insert_neo4j_data(line):
	def get_items(line):
		if '{' not in line and '}' not in line:
			return [line]
		# 检查
		if '{' not in line or '}' not in line:
			print('get_items Error', line)
		lines = [w[1:-1] for w in re.findall('{.*?}', line)]
		return lines
		# 查找节点是否不存，不存在就创建一个

	def look_and_create(name):

		end = graph.find_one(label="car_industry", property_key="name", property_value=name)
		if end == None:
			end = Node('car_industry', name=name)
		return end
	# -----------insert_one_data-------------
	if '' in line:
		print('insert_one_data错误', line)
		return ""

	start = look_and_create(line[0])
	for name in get_items(line[2]):
		end = look_and_create(name)
		r = Relationship(start, line[1], end, name=line[1])
		graph.create(r)  # 当存在时不会创建新的
	q = "MATCH (n)\n" + "RETURN count(n)"
	sum_graph = list(graph.data(q))[0]["count(n)"]
	print("sum_graph:", sum_graph)
	return sum_graph

def ner(request):
	sentence = request.GET.get("sentence", "")
	print("sentence:", sentence)
	result_word = []

	result_tmp = main_lstm.predict(sentence)
	for v in result_tmp:
		if v != "":
			result_word.append(v)
	q = "MATCH (n)\n" + "RETURN count(n)"
	old_sum_graph = list(graph.data(q))[0]["count(n)"]
	print("result:", old_sum_graph)
	if len(result_word) >= 2:
		w1 = result_word[0]
		w2 = result_word[1]
		rela = predict_rela.pred_rela("{}##{}##{}".format(w1, w2, sentence))
		rela_neo4j = rela.split(",")[0].split(":")[0]
		print("rela_neo4j:",rela_neo4j)
		new_sum_graph = insert_neo4j_data([w1, rela_neo4j, w2])
		if new_sum_graph=="":
			new_sum_graph = old_sum_graph
	else:
		rela = 'please check sentence!'
		new_sum_graph = old_sum_graph

	return render(request, 'index_ner.html', {'ner': ','.join(result_word),
											  'sentence': sentence, 'rela': rela,"old_num":old_sum_graph,"new_num":new_sum_graph})

def relation(request):
	ctx = {}
	if(request.GET):
		entity1 = request.GET['entity1_text']
		relation = request.GET['relation_name_text']
		entity2 = request.GET['entity2_text']
		# print('entity1=',entity1)
		# print('relation=', relation)
		# print('entity2=', entity2)
		relation = relation.lower()
		searchResult = {}
		#若只输入entity1,则输出与entity1有直接关系的实体和关系
		if(len(entity1) != 0 and len(relation) == 0 and len(entity2) == 0):
			searchResult = findRelationByEntity(entity1)
			print('searchResult=', searchResult)
			print('type(searchResult)=', type(searchResult))
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		#若只输入entity2则,则输出与entity2有直接关系的实体和关系
		if(len(entity2) != 0 and len(relation) == 0 and len(entity1) == 0):
			searchResult = findRelationByEntity2(entity2)

			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})
		#若输入entity1和relation，则输出与entity1具有relation关系的其他实体
		if(len(entity1)!=0 and len(relation)!=0 and len(entity2) == 0):
			searchResult = findOtherEntities(entity1,relation)

			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})
		#若输入entity2和relation，则输出与entity2具有relation关系的其他实体
		if(len(entity2)!=0 and len(relation)!=0 and len(entity1) == 0):
			searchResult = findOtherEntities2(entity2,relation)

			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})
		#若输入entity1和entity2,则输出entity1和entity2之间的最短路径
		if(len(entity1) !=0 and len(relation) == 0 and len(entity2)!=0):

			searchResult = findRelationByEntities(entity1,entity2)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})
		#若输入entity1,entity2和relation,则输出entity1、entity2是否具有相应的关系
		if(len(entity1)!=0 and len(entity2)!=0 and len(relation)!=0):
			searchResult = findEntityRelation(entity1,relation,entity2)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})
		#全为空
		if(len(entity1)!=0 and len(relation)!=0 and len(entity2)!=0 ):
			pass
		ctx= {'title' : '<h1>暂未找到相应的匹配</h1>'}
		return render(request,'relation.html',{'ctx':ctx})
	return render(request, 'relation.html', {'ctx': ctx})


if __name__ == '__main__':
	data = predict_rela.pred_rela("阻抗角##容性元件##当阻抗角小于0时，元件为容性元件")
	print(data)
