### Demidifyer mulltiple
#!/bin/bash

for i in {2,3,10,21,29,30,31}
do
   experiment='paper_runs'
   fmtransfer -c cfg/${experiment}.yaml \
      --data.data_file=${experiment}_$i.pt \
      --data.generator.patch_loc=$i \
      --trainer.callbacks+=pytorch_lightning.callbacks.ModelCheckpoint \
      --trainer.callbacks.dirpath='./logs/checkpoints/'${experiment}'/' \
      --trainer.callbacks.filename=patch_$i \
      --trainer.callbacks.monitor='validation/loss' \
      --trainer.callbacks.verbose=true \
      --trainer.logger.name=patch_$i
      #--trainer.max_steps=100 \
      #--trainer.fast_dev_run=False
done
