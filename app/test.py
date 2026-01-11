from langfuse import Langfuse
from dotenv import load_dotenv
import inspect

load_dotenv()
lf = Langfuse()

# print(dir(lf))
# print(inspect.signature(lf.create_event))
# print(lf.create_event.__doc__)
# print(inspect.getsource(lf.create_event))
help(lf.create_event)
