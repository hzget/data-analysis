"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc

import analysis_pb2
import analysis_pb2_grpc

import bow

class DataAnalysis(analysis_pb2_grpc.DataAnalysisServicer):

    def AnalyzeByAuthor(self, request, context):
        logging.info(f'receive msg: {request.name}')
        return analysis_pb2.Status(score=2)

    def AnalyzeByPostId(self, request, context):
        logging.info(f'receive msg: {request.id}')
        return analysis_pb2.PostResult(result=str(request.id))

    def AnalyzePost(self, request, context):
        logging.info(f'receive msg: {request.text}')
        return analysis_pb2.Text(text=bow.getTextClass(request.text))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    analysis_pb2_grpc.add_DataAnalysisServicer_to_server(DataAnalysis(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # load AI model
    logging.info(f'loading AI model ...')
    bow.load()

    # start server
    logging.info(f'starting service ...')
    serve()
