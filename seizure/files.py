try:
    import cPickle as pickle
except:
    import pickle
import glob
import json
import logging
import os
import scipy.io
import sqlite3
import unicodedata

class FileContext(object):
    """
    Stores the location of the directories for data files, cache files, and submission files.
    """
    def __init__(self, data_dir, cache_dir, report_dir, submission_dir, log_file, limit_targets=0):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.report_dir = report_dir
        self.submission_dir = submission_dir
        self.log_file = log_file
        self.database = Database(os.path.join(cache_dir,'cache.db'))
        self.limit = None #Limit for number of clips
        
        # Make dirs for cache, report, and submission
        if not os.path.exists(submission_dir):
            os.makedirs(submission_dir)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        def _get_data_dirs(query):
            dirs = glob.glob(os.path.join(data_dir, "*%s*" % query))
            if dirs == []:
                err = ValueError("%s data files cannot be found in %s" % (query, data_dir))
                logging.exception(err)
                raise err
            else:
                return dirs
           
        self.dog_dirs = _get_data_dirs("Dog")
        self.patient_dirs = _get_data_dirs("Patient")
        self.targets = self.dog_dirs + self.patient_dirs
        if limit_targets:
            self.targets = self.targets[:limit_targets]
            
    def get_preictal_segments(self, *targets):
        return self.search_files(targets, '*preictal_segment*')
    
    def get_interictal_segments(self, *targets):
        return self.search_files(targets, '*interictal_segment*')
            
    def get_test_segments(self, *targets):
        return self.search_files(targets, '*test_segment*')
            
    def search_files(self, targets, query):
        filenames = []
        for target in targets:
            filenames += glob.glob(os.path.join(target,query))
        if self.limit != None:
            return filenames[0:min(self.limit,len(filenames))]
        return filenames
    
    def cache_dump(self, data, tag, extra=None):
        dict_ = pickle.dumps(data.__dict__) if hasattr(data, '__dict__') else 'None'
        self.database.dump(str(data.__class__), tag, pickle.dumps(data), dict_, extra)
    
    def cache_load(self, tag, extra=None, cls=None):
        data = self.database.load(str(cls), tag, extra)
        if data != None:
            dump, dict_ = data
            instance = pickle.loads(dump)
            if dict_ != 'None':
                instance.__dict__ = pickle.loads(dict_)
            return instance
        
    def cache_delete(self, tag, extra=None, cls=None):
        self.database.delete(str(cls), str(tag), str(extra))
    
    def set_limit(self, num):
        self.limit = num
            
    @staticmethod
    def from_json(json_file, limit_targets=0):
        
        def _abspath(dirname):
            return dirname if os.path.isabs(dirname) else os.path.join(os.path.dirname(json_file), dirname)
        
        with open(json_file) as f:
            settings = json.load(f)
            
        data_dir = _abspath(str(settings['competition-data-dir']))
        cache_dir = _abspath(str(settings['data-cache-dir']))
        report_dir = _abspath(str(settings['report-dir']))
        submission_dir = _abspath(str(settings['submission-dir']))
        log_file = _abspath(str(settings['log-file']))
           
        return FileContext(data_dir, cache_dir, report_dir, submission_dir, log_file, limit_targets)

class Database(object):
    """
    Stores data cache within a database using a unique three element key pair.
    """
    def __init__(self, db_filename, num_conn=4):
        self.filename = db_filename
        self.conns = [sqlite3.connect(self.filename) for __ in range(0, num_conn)]
        self.add_data_table()
        
    def get(self):
        try:
            return self.conns.pop()
        except IndexError:
            return sqlite3.connect(self.filename)
    
    def put(self, conn):
        self.conns.append(conn)
        
    def add_file_table(self):
        table = """
        CREATE TABLE IF NOT EXISTS clips (
            target TEXT NOT NULL,
            clip TEXT NOT NULL,
            type TEXT NOT NULL,
            PRIMARY KEY(clip)
        );
        """
        db = self.get()
        c = db.cursor()
        c.execute(table)
        db.commit()
        self.put(db)
        
    def add_data_table(self):
        table = """
        CREATE TABLE IF NOT EXISTS objs (
            class TEXT NOT NULL,
            tag TEXT NOT NULL,
            dump TEXT NOT NULL,
            dict TEXT NOT NULL,
            extra TEXT NOT NULL,
            PRIMARY KEY(class, tag, extra)
        );
        """
        db = self.get()
        c = db.cursor()
        c.execute(table)
        db.commit()
        self.put(db)
        
    def dump(self, cls, tag, dump, dict_, extra):
        db = self.get()
        ucode = unicode(dump, "utf-8")
        dict_ = unicode(dict_, "utf-8")
        c = db.cursor()
        c.execute("INSERT OR REPLACE INTO objs(class, tag, dump, dict, extra) VALUES (?, ?, ?, ?, ?)", (cls, tag, ucode, dict_, extra))
        db.commit()
        self.put(db)
    
    def load(self, cls, tag, extra):
        db = self.get()
        c = db.cursor()
        if cls == "None":
            c.execute("SELECT dump, dict FROM objs WHERE tag=? AND extra=?", (tag, extra))
        else:
            c.execute("SELECT dump, dict FROM objs WHERE class=? AND tag=? AND extra=?", (cls, tag, extra))
        result = c.fetchone()
        self.put(db)
        if result:
            return result[0].encode("utf-8"), result[1].encode("utf-8")
        else:
            return None
    
    def delete(self, cls, tag, extra):
        db = self.get()
        c = db.cursor()
        c.execute("DELETE FROM objs WHERE class=? AND tag=? AND extra=?", (cls, tag, extra))
        db.commit()
        self.put(db)
    
    def reset(self):
        db = self.get()
        c = db.cursor()
        c.execute("DROP TABLE objs")
        db.commit()
        self.put(db)
        self.add_data_table()