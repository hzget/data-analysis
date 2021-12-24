"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc

import analysis_pb2
import analysis_pb2_grpc


class DataAnalysis(analysis_pb2_grpc.DataAnalysisServicer):

    def AnalyzeByAuthor(self, request, context):
        logging.info(f'receive msg: {request.name}')
        return analysis_pb2.Status(score=2)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    analysis_pb2_grpc.add_DataAnalysisServicer_to_server(DataAnalysis(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()