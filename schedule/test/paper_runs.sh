### Demidifyer mulltiple
#!/bin/bash
echo Running Test Script . . .
for i in {2,3,10,21,29,30,31}
do
   experiment='paper_runs'
   fmtransfer test -c cfg/${experiment}.yaml \
      --data.data_file=${experiment}_$i.pt \
      --data.generator.patch_loc=$i \
      --ckpt_path=logs/checkpoints/${experiment}/patch_$i.ckpt \
      --trainer.logger.name=patch_$i
done
