import sys
import time
from colorama import Fore, Back, Style


PROGRESS_BAR_LENGTH = 40

# ProgressBar, by Michael Stewart.
# A simple progress bar that should be called every epoch.
class ProgressBar():
	def __init__(self, num_batches, max_epochs, logger):
		self.num_batches = num_batches
		self.max_epochs = max_epochs
		self.logger = logger

	def draw_bar(self, i, epoch, epoch_start_time):
		epoch_progress = int(PROGRESS_BAR_LENGTH * (1.0 * i / self.num_batches))
		sys.stdout.write("      Epoch %s of %d : [" % (str(epoch).ljust(len(str(self.max_epochs))), self.max_epochs))
		sys.stdout.write("=" * epoch_progress)
		sys.stdout.write(">")
		sys.stdout.write(" " * (PROGRESS_BAR_LENGTH - epoch_progress))
		sys.stdout.write("] %d / %d" % (i + 1, self.num_batches))
		sys.stdout.write("   %.2fs" % (time.time() - epoch_start_time))
		sys.stdout.write("\r")
		sys.stdout.flush()

	def draw_completed_epoch(self, loss, loss_list, epoch, epoch_start_time, f1_score=None):		
		outstr =  ""
		outstr += "Epoch %s of %d : Loss = %.6f" % (str(epoch).ljust(len(str(self.max_epochs))), self.max_epochs, loss)
		if epoch > 1:
			diff = loss - loss_list[epoch - 2]
			col = Fore.GREEN if diff < 0 else "%s+" % Fore.RED
			outstr += " %s%.6f%s" % ( col, loss - loss_list[epoch - 2], Style.RESET_ALL )
		else:
			outstr += " " * 10
		if f1_score is not None:
			outstr += "   F1: %.4f" % f1_score
		else:
			outstr += "             "
		outstr += "   %.2fs" % (time.time() - epoch_start_time)
		self.logger.info("%s%s" % (outstr, " " * (PROGRESS_BAR_LENGTH - len(outstr) + 60)))
		sys.stdout.flush()
				
		
		
