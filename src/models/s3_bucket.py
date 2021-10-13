import boto3
import pandas as pd
import os

os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
os.environ["AWS_ACCESS_KEY_ID"] = 'AKIAZURDHJ7XTWF2Q55F'
os.environ["AWS_SECRET_ACCESS_KEY"] = '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'


class S3Bucket(object):
    def __init__(self) -> None:
        super().__init__()
        self.s3 = boto3.resource(
            service_name='s3',
            region_name='us-east-1',
            aws_access_key_id='AKIAZURDHJ7XTWF2Q55F',
            aws_secret_access_key='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'
        )
        self.s3_name = 'rganti-qqq-data'

    def push_to_s3(self, input_path, output_path):
        print("Pushing {} to {}".format(input_path, self.s3_name))
        self.s3.Bucket(self.s3_name).upload_file(Filename=input_path, Key=output_path)

    def load_from_s3(self, file_name, index=False):
        obj = self.s3.Bucket(self.s3_name).Object(file_name).get()
        if index:
            df = pd.read_csv(obj['Body'])
        else:
            df = pd.read_csv(obj['Body'], index_col=0)
        return df
