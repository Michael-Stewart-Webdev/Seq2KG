''' logger '''
import logging as logger
import codecs
import sys, torch, os
from datetime import datetime
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGFILE = None

class LoggingFormatter(logger.Formatter):	

	def format(self, record):
		#compute s according to record.levelno
		#for example, by setting self._fmt
		#according to the levelno, then calling
		#the superclass to do the actual formatting
			
		if record.levelno == 10:
			return Fore.CYAN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL,  record.msg)
		elif record.levelno == 20:
			return Fore.GREEN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg) 
		elif record.levelno == 30:
			return Fore.YELLOW + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg)
		else:
			return Fore.RED + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg)
		#return s

class LogToFile(logger.Formatter):
	def format(self, record):

		# Break the message nicely onto multiple lines so the file looks a bit better.
		def break_lines(msg):
			def get_chunks(msg, n=80):
				chunks = []
				for i in range(0, len(msg), n):
					chunks.append(msg[i: i + n])
				return chunks
			lines = get_chunks(msg.strip())
			for i in range(1, len(lines)):
				lines[i] = (" " * 28) + lines[i]
			return "\n".join(lines)

		message = record.msg.replace(Fore.GREEN, "")
		message = message.replace(Fore.RED, "")
		message = message.replace(Fore.YELLOW, "")
		message = message.replace(Style.RESET_ALL, "")

		return "%s %s %s" % (datetime.now().strftime('%d-%m-%Y %H:%M:%S'), record.levelname.ljust(7), message)

hdlr = logger.StreamHandler(sys.stdout)
hdlr.setFormatter(LoggingFormatter())
logger.root.addHandler(hdlr)
logger.root.setLevel(logger.DEBUG)