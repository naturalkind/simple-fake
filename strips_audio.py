import bpy
import time
### Strips audio ###
for scene in bpy.data.scenes: 
    if scene.sequence_editor is not None:
        strip = scene.sequence_editor.sequences_all
        if strip is not None:
            bpy.ops.sequencer.select_all(action='DESELECT')  
            for ix, i in enumerate(strip):
                    i.select = True
                    temp = i.frame_final_end-i.frame_final_duration
                    bpy.context.scene.frame_start = temp
                    bpy.context.scene.frame_end = i.frame_final_end
                    bpy.ops.sound.mixdown(filepath=f"//cutaudio/audio/a{ix}.wav", 
                                          relative_path=True, 
                                          container='WAV', codec='PCM')
                    time.sleep(3)
                    ix += 1
                    bpy.ops.sequencer.select_all(action='DESELECT')
