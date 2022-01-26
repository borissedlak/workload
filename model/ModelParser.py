import ast

from Transformations import Blur_Face_Pixelate, Max_Spec_Resize
from Triggers import Face_Trigger, Age_Trigger

source_commands = ['video', 'audio']
trigger_functions = ['Face_Trigger', 'Age_Trigger']
transformation_functions = ['Blur_Face_Pixelate', 'Max_Spec_Resize']
triggers_and_transformations = trigger_functions + transformation_functions


class CmdWithArgs:

    def __init__(self, s: str):
        self.command = s.split(':', 1)[0]
        self.args = ast.literal_eval(s.split(':', 1)[1])
        self.commandFunction = None

    def resolveCommand(self):
        self.commandFunction = {
            'Face_Trigger': Face_Trigger(),
            'Age_Trigger': Age_Trigger(),
            'Blur_Face_Pixelate': Blur_Face_Pixelate(),
            'Max_Spec_Resize': Max_Spec_Resize()
        }.get(self.command)

    def isMediaSource(self):
        return self.command in source_commands

    def isTrigger(self):
        return self.command in trigger_functions

    def isTransformation(self):
        return self.command in transformation_functions


class PrivacyChain:

    def __init__(self, cmA: list[CmdWithArgs]):
        self.mediaSource = cmA[0]
        self.cmAs = cmA[1:]

    def printInfo(self):

        print(f"======== Media Source ========")
        print(f"Source: {self.mediaSource.command}")
        print(f"Arguments: {self.mediaSource.args}\n")

        for idx, val in enumerate(self.cmAs):
            print(f"========= {idx + 1}. Step =========")
            print(f"Command: {val.command}")
            if val.isTrigger():
                print(f"Trigger Function")
            elif val.isTransformation():
                print(f"Transformation Function")
            print(f"Arguments: {val.args}")
        print("\n")


class PrivacyModel:

    def __init__(self, chains: list[PrivacyChain]):
        self.chains = chains

    # If a tag is here, this means that the provider has tagged his stream, and we should try to find a specific model
    def getChainForSource(self, media_type, tag=None):
        match_filter = list(filter(lambda c: c.mediaSource.command == media_type, self.chains))
        if len(match_filter) <= 0:
            raise ValueError("The incoming stream has an invalid media type or there is no model present to allow it")

        if tag is not None:
            filtered_tag = list(
                filter(lambda c: 'tag' in c.mediaSource.args and c.mediaSource.args['tag'] == tag,
                       match_filter))
            if len(filtered_tag) <= 0:
                print(f"No privacy chain in the model was tagged with {tag}, using untagged chains")
            else:
                match_filter = filtered_tag
        return match_filter[-1]


def parseModel(s: str):
    chains: list[PrivacyChain] = []
    chains_raw: list[str] = s.splitlines()

    for c in chains_raw:

        commandsWithArgs = []
        cAs = c.split('-->')

        for cA in cAs:
            commandsWithArgs.append(CmdWithArgs(cA))

        if not commandsWithArgs[0].isMediaSource():
            raise ValueError(f"First Command '{commandsWithArgs[0].command}' does not specify a media source")

        for idx, val in enumerate(commandsWithArgs[1:]):
            if val.command not in triggers_and_transformations:
                raise ValueError(f"Command '{val.command}' unknown")
            commandsWithArgs[idx + 1].resolveCommand()

        chains.append(PrivacyChain(commandsWithArgs))

    return PrivacyModel(chains)

# model = parseModel(Models.faces_pixelate_with_resize)
# # model = PrivacyModel([chain])
# print(model.getChainForSource("video", "webcam"))
