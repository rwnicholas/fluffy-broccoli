import requests

class RequestSEFAZ:
	'''
	Request from SEFAZ Economiza Alagoas
	'''
	def __init__(self):
		super().__init__()
	
	def request(self, term):
		r = requests.post("http://api.sefaz.al.gov.br/sfz_nfce_api/api/public/consultarPrecosPorDescricao",
			headers={
				"Content-Type": "application/json",
				"AppToken": "d006e4751057f5dd9c0a174ebc108c292271797a"
			},
			json={
				"descricao": term,
				"dias": 3,
				"latitude": -9.6432331,
				"longitude": -35.7190686,
				"raio": 15
			}
		)
		return r

class RequestGet():
	'''
	Request data, useful with Portal de Compras Governamentais
	'''
	count = None
	returnedData = None

	def __init__(self, params = None, url = None):
		if url != None:
			self.returnedData = requests.get(url=url, params=params).json()
			self.count = self.returnedData['count']
		
	def request(self, params, url):
		self.returnedData = requests.get(url=url, params=params).json()
