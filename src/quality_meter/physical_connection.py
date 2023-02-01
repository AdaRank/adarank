

class PhysicalConnection:
    def __init__(self, source_id, target_id, source_mlfb, target_mlfb, id_, anonymized_source_id, anonymized_target_id):
        self.id = id_
        self.target_id = target_id
        self.source_id = source_id
        self.target_mlfb = target_mlfb
        self.source_mlfb = source_mlfb
        self.anonymized_source_id = anonymized_source_id
        self.anonymized_target_id = anonymized_target_id

    def equals(self, comp):
        if self.anonymized_source_id == comp.anonymized_source_id and self.anonymized_target_id == comp.anonymized_target_id:
            return True
        else:
            return False

    def get_anonymized_source_id(self):
        return self.anonymized_source_id

    def get_anonymized_target_id(self):
        return self.anonymized_target_id

    def __str__(self):
        return f"Physical connection consisting of source: {self.anonymized_source_id} and target: {self.get_anonymized_target_id()}"